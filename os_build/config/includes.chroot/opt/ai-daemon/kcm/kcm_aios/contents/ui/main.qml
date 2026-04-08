import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kcmutils as KCM
import org.kde.kirigami as Kirigami

/*
 * KCM: AI Assistant (System Settings → AI Assistant)
 *
 * Tabbed config UI:
 *   Model      — list local GGUFs, switch, load, unload
 *   Cloud      — OpenRouter toggle, API key via KWallet, test
 *   Inference  — generation param sliders (hot-applied)
 *   Behaviour  — autostart, lazy-load, notification prefs
 *   About      — live status, version
 *
 * Writes to ~/.config/ai-daemon/settings.yaml via the `aios-settings-write`
 * helper, then calls ReloadSettings() on the daemon.
 */
KCM.SimpleKCM {
    id: kcm

    title: i18n("AI Assistant")

    // Shared state ─────────────────────────────────────────────────────────
    property var daemonStatus: ({
        backend: "offline",
        model: "",
        model_loaded: false,
        loading: false,
        ram_used_mb: 0,
        vram_used_mb: 0,
        vram_free_mb: 0,
        version: "",
        uptime_seconds: 0,
    })
    property var models: []
    // Draft of settings currently displayed. Any change marks the KCM dirty
    // so System Settings enables the Apply button via `needsSave`.
    property var draft: ({
        model: { default_local: "", lazy_load: true, auto_start: false },
        generation: {
            temperature: 0.5, top_p: 0.9, top_k: 40,
            repeat_penalty: 1.1, max_tokens: 1024,
            n_ctx: 2048, n_gpu_layers: -1, seed: -1,
        },
        openrouter: { enabled: false, default_model: "deepseek/deepseek-chat" },
        behaviour: { notify_on_moderate: false, vram_threshold_mb: 500 },
    })
    property bool dirty: false
    property string openRouterKey: ""        // held only for the session
    property string lastMessage: ""

    DaemonBridge {
        id: bridge
        onStatusReceived: function(status) { kcm.daemonStatus = status; }
        onModelsReceived: function(models) { kcm.models = models; }
        onOperationResult: function(op, ok, msg) {
            kcm.lastMessage = ok ? (op + ": ok") : (op + ": " + msg);
            if (op === "load" || op === "unload" || op === "reload"
                || op === "switch" || op === "autostart")
                bridge.fetchStatus();
        }
    }

    Timer {
        id: poll
        interval: 2000
        running: true
        repeat: true
        triggeredOnStart: true
        onTriggered: {
            bridge.fetchStatus();
            if (kcm.models.length === 0) bridge.fetchModels();
        }
    }

    Component.onCompleted: {
        bridge.fetchStatus();
        bridge.fetchModels();
    }

    // System Settings "Apply" button glue ─────────────────────────────────
    // `kcm.needsSave` drives the Apply/Reset buttons. We set it from `dirty`.
    onDraftChanged: dirty = true

    // SimpleKCM exposes save() / load() hooks — we use custom buttons
    // per-tab instead since operations are varied (save vs. load model vs.
    // test connection) and SimpleKCM's single Apply doesn't fit them all.

    ColumnLayout {
        Layout.fillWidth: true
        spacing: Kirigami.Units.largeSpacing

        // Status banner (always visible) ──────────────────────────────────
        Kirigami.InlineMessage {
            Layout.fillWidth: true
            visible: true
            type: daemonStatus.backend === "offline" ? Kirigami.MessageType.Warning
                  : daemonStatus.loading ? Kirigami.MessageType.Information
                  : daemonStatus.model_loaded ? Kirigami.MessageType.Positive
                  : Kirigami.MessageType.Information
            text: {
                if (daemonStatus.backend === "offline")
                    return i18n("Daemon offline. Start with: systemctl --user start ai-daemon");
                if (daemonStatus.loading)
                    return i18n("Loading model…");
                if (daemonStatus.model_loaded)
                    return i18n("Model loaded: %1 (%2, RAM %3 MB)",
                                daemonStatus.model,
                                daemonStatus.backend.toUpperCase(),
                                daemonStatus.ram_used_mb);
                return i18n("Daemon running — model not loaded (lazy-load is on). "
                           + "The next query will load it.");
            }
        }

        QQC2.TabBar {
            id: tabs
            Layout.fillWidth: true
            QQC2.TabButton { text: i18n("Model") }
            QQC2.TabButton { text: i18n("Cloud") }
            QQC2.TabButton { text: i18n("Inference") }
            QQC2.TabButton { text: i18n("Behaviour") }
            QQC2.TabButton { text: i18n("About") }
        }

        StackLayout {
            Layout.fillWidth: true
            currentIndex: tabs.currentIndex

            // ── 1. Model ────────────────────────────────────────────────
            Kirigami.FormLayout {
                id: modelPage

                QQC2.Label {
                    Kirigami.FormData.label: i18n("Active model:")
                    text: daemonStatus.model || i18n("(none)")
                }
                QQC2.Label {
                    Kirigami.FormData.label: i18n("State:")
                    text: daemonStatus.loading ? i18n("Loading…")
                          : daemonStatus.model_loaded
                            ? i18n("Loaded  ·  RAM %1 MB  ·  VRAM %2 / %3 MB",
                                   daemonStatus.ram_used_mb,
                                   daemonStatus.vram_used_mb,
                                   daemonStatus.vram_used_mb + daemonStatus.vram_free_mb)
                            : i18n("Not loaded")
                }

                QQC2.ComboBox {
                    id: modelBox
                    Kirigami.FormData.label: i18n("Select model:")
                    Layout.fillWidth: true
                    model: kcm.models.filter(function(m) { return m.type === "gguf"; })
                                     .map(function(m) { return m.name; })
                }

                RowLayout {
                    Kirigami.FormData.label: ""
                    spacing: Kirigami.Units.smallSpacing

                    QQC2.Button {
                        text: i18n("Load / Switch")
                        enabled: modelBox.currentText !== ""
                                 && !daemonStatus.loading
                        onClicked: bridge.loadModel(modelBox.currentText)
                    }
                    QQC2.Button {
                        text: i18n("Unload")
                        enabled: daemonStatus.model_loaded && !daemonStatus.loading
                        onClicked: unloadConfirm.open()
                    }
                    QQC2.Button {
                        text: i18n("Refresh")
                        onClicked: bridge.fetchModels()
                    }
                }

                QQC2.Label {
                    Kirigami.FormData.label: ""
                    text: i18n("Drop additional GGUF files into /opt/ai-daemon/models/ "
                               + "and click Refresh.")
                    font: Kirigami.Theme.smallFont
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    color: Kirigami.Theme.disabledTextColor
                }
            }

            // ── 2. Cloud ────────────────────────────────────────────────
            Kirigami.FormLayout {
                id: cloudPage

                QQC2.CheckBox {
                    id: cloudEnabled
                    Kirigami.FormData.label: i18n("OpenRouter:")
                    text: i18n("Enable cloud inference")
                    checked: kcm.draft.openrouter.enabled
                    onToggled: {
                        kcm.draft.openrouter.enabled = checked;
                        kcm.dirty = true;
                    }
                }

                QQC2.TextField {
                    id: cloudKey
                    Kirigami.FormData.label: i18n("API key:")
                    Layout.fillWidth: true
                    echoMode: TextInput.Password
                    placeholderText: i18n("sk-or-…")
                    text: kcm.openRouterKey
                    onEditingFinished: kcm.openRouterKey = text
                }

                QQC2.TextField {
                    id: cloudModel
                    Kirigami.FormData.label: i18n("Model ID:")
                    Layout.fillWidth: true
                    text: kcm.draft.openrouter.default_model
                    placeholderText: "deepseek/deepseek-chat"
                    onEditingFinished: {
                        kcm.draft.openrouter.default_model = text;
                        kcm.dirty = true;
                    }
                }

                RowLayout {
                    Kirigami.FormData.label: ""
                    QQC2.Button {
                        text: i18n("Save key")
                        enabled: cloudKey.text !== ""
                        onClicked: {
                            kcm.openRouterKey = cloudKey.text;
                            bridge.setOpenRouterKey(cloudKey.text);
                        }
                    }
                    QQC2.Button {
                        text: i18n("Test connection")
                        enabled: cloudKey.text !== ""
                        onClicked: bridge.testOpenRouter(cloudKey.text, cloudModel.text)
                    }
                    QQC2.Button {
                        text: i18n("Switch to cloud")
                        enabled: kcm.draft.openrouter.enabled
                                 && cloudModel.text !== ""
                        onClicked: bridge.switchModel(
                            "openrouter:" + cloudModel.text)
                    }
                }

                QQC2.Label {
                    Kirigami.FormData.label: ""
                    text: i18n("The key is stored in KWallet and mirrored to "
                               + "~/.config/ai-daemon/secrets.yaml (mode 0600) so "
                               + "the daemon can read it at runtime.")
                    font: Kirigami.Theme.smallFont
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    color: Kirigami.Theme.disabledTextColor
                }
            }

            // ── 3. Inference ────────────────────────────────────────────
            Kirigami.FormLayout {
                id: inferencePage

                QQC2.Slider {
                    id: tempSlider
                    Kirigami.FormData.label: i18n("Temperature:")
                    from: 0.0; to: 2.0; stepSize: 0.05
                    value: kcm.draft.generation.temperature
                    onMoved: {
                        kcm.draft.generation.temperature = Math.round(value*100)/100;
                        kcm.dirty = true;
                    }
                }
                QQC2.Label { text: tempSlider.value.toFixed(2) }

                QQC2.Slider {
                    id: topPSlider
                    Kirigami.FormData.label: i18n("Top-p:")
                    from: 0.0; to: 1.0; stepSize: 0.01
                    value: kcm.draft.generation.top_p
                    onMoved: {
                        kcm.draft.generation.top_p = Math.round(value*100)/100;
                        kcm.dirty = true;
                    }
                }
                QQC2.Label { text: topPSlider.value.toFixed(2) }

                QQC2.SpinBox {
                    id: topKSpin
                    Kirigami.FormData.label: i18n("Top-k (0 = off):")
                    from: 0; to: 200
                    value: kcm.draft.generation.top_k
                    onValueModified: {
                        kcm.draft.generation.top_k = value;
                        kcm.dirty = true;
                    }
                }

                QQC2.Slider {
                    id: repeatSlider
                    Kirigami.FormData.label: i18n("Repeat penalty:")
                    from: 1.0; to: 1.5; stepSize: 0.01
                    value: kcm.draft.generation.repeat_penalty
                    onMoved: {
                        kcm.draft.generation.repeat_penalty = Math.round(value*100)/100;
                        kcm.dirty = true;
                    }
                }
                QQC2.Label { text: repeatSlider.value.toFixed(2) }

                QQC2.SpinBox {
                    id: maxTokSpin
                    Kirigami.FormData.label: i18n("Max output tokens:")
                    from: 64; to: 8192; stepSize: 64
                    value: kcm.draft.generation.max_tokens
                    onValueModified: {
                        kcm.draft.generation.max_tokens = value;
                        kcm.dirty = true;
                    }
                }

                QQC2.ComboBox {
                    id: ctxBox
                    Kirigami.FormData.label: i18n("Context window:")
                    model: [1024, 2048, 4096, 8192]
                    currentIndex: Math.max(0, model.indexOf(kcm.draft.generation.n_ctx))
                    onActivated: {
                        kcm.draft.generation.n_ctx = model[currentIndex];
                        kcm.dirty = true;
                    }
                }

                QQC2.SpinBox {
                    id: gpuLayersSpin
                    Kirigami.FormData.label: i18n("GPU layers (-1 = all):")
                    from: -1; to: 200
                    value: kcm.draft.generation.n_gpu_layers
                    onValueModified: {
                        kcm.draft.generation.n_gpu_layers = value;
                        kcm.dirty = true;
                    }
                }

                QQC2.SpinBox {
                    id: seedSpin
                    Kirigami.FormData.label: i18n("Seed (-1 = random):")
                    from: -1; to: 2147483647
                    value: kcm.draft.generation.seed
                    onValueModified: {
                        kcm.draft.generation.seed = value;
                        kcm.dirty = true;
                    }
                }

                Kirigami.InlineMessage {
                    Layout.fillWidth: true
                    visible: true
                    type: Kirigami.MessageType.Information
                    text: i18n("Context window and GPU layers are reload-required. "
                               + "Other parameters apply on the next query.")
                }

                RowLayout {
                    Kirigami.FormData.label: ""
                    QQC2.Button {
                        text: i18n("Apply")
                        enabled: kcm.dirty
                        onClicked: applySettings()
                    }
                    QQC2.Button {
                        text: i18n("Reset to defaults")
                        onClicked: resetToDefaults()
                    }
                }
            }

            // ── 4. Behaviour ────────────────────────────────────────────
            Kirigami.FormLayout {
                id: behaviourPage

                QQC2.CheckBox {
                    text: i18n("Auto-start daemon on login")
                    checked: kcm.draft.model.auto_start
                    onToggled: {
                        kcm.draft.model.auto_start = checked;
                        kcm.dirty = true;
                        bridge.setAutoStart(checked);
                    }
                }
                QQC2.CheckBox {
                    text: i18n("Lazy-load model (defer until first query)")
                    checked: kcm.draft.model.lazy_load
                    onToggled: {
                        kcm.draft.model.lazy_load = checked;
                        kcm.dirty = true;
                    }
                }
                QQC2.CheckBox {
                    text: i18n("Notify on MODERATE-risk commands")
                    checked: kcm.draft.behaviour.notify_on_moderate
                    onToggled: {
                        kcm.draft.behaviour.notify_on_moderate = checked;
                        kcm.dirty = true;
                    }
                }

                QQC2.SpinBox {
                    Kirigami.FormData.label: i18n("VRAM warning threshold (MB):")
                    from: 0; to: 4096; stepSize: 50
                    value: kcm.draft.behaviour.vram_threshold_mb
                    onValueModified: {
                        kcm.draft.behaviour.vram_threshold_mb = value;
                        kcm.dirty = true;
                    }
                }

                QQC2.Button {
                    Kirigami.FormData.label: ""
                    text: i18n("Apply")
                    enabled: kcm.dirty
                    onClicked: applySettings()
                }
            }

            // ── 5. About ────────────────────────────────────────────────
            Kirigami.FormLayout {
                id: aboutPage

                QQC2.Label {
                    Kirigami.FormData.label: i18n("Daemon version:")
                    text: daemonStatus.version || i18n("(offline)")
                }
                QQC2.Label {
                    Kirigami.FormData.label: i18n("Uptime:")
                    text: formatUptime(daemonStatus.uptime_seconds)
                }
                QQC2.Label {
                    Kirigami.FormData.label: i18n("Backend:")
                    text: daemonStatus.backend.toUpperCase()
                }
                QQC2.Label {
                    Kirigami.FormData.label: i18n("RAM used:")
                    text: daemonStatus.ram_used_mb + " MB"
                }
                QQC2.Label {
                    Kirigami.FormData.label: i18n("VRAM used:")
                    text: (daemonStatus.vram_used_mb + " / "
                           + (daemonStatus.vram_used_mb + daemonStatus.vram_free_mb)
                           + " MB")
                }
                QQC2.Label {
                    Kirigami.FormData.label: i18n("Last message:")
                    text: kcm.lastMessage
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                }
            }
        }
    }

    // ── Unload confirm ─────────────────────────────────────────────────────
    QQC2.Dialog {
        id: unloadConfirm
        title: i18n("Unload model?")
        standardButtons: QQC2.Dialog.Ok | QQC2.Dialog.Cancel
        modal: true
        onAccepted: bridge.unloadModel()

        contentItem: QQC2.Label {
            wrapMode: Text.Wrap
            text: i18n("This frees the model from RAM and VRAM. "
                       + "The next query will reload it "
                       + "(typically 3–5 s on GPU, longer on CPU). Continue?")
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    function formatUptime(s) {
        if (!s) return "—";
        var h = Math.floor(s / 3600);
        var m = Math.floor((s % 3600) / 60);
        return (h > 0 ? h + "h " : "") + m + "m";
    }

    function applySettings() {
        bridge.saveSettings(JSON.stringify(kcm.draft));
        kcm.dirty = false;
    }

    function resetToDefaults() {
        kcm.draft = {
            model: { default_local: kcm.draft.model.default_local,
                     lazy_load: true, auto_start: false },
            generation: {
                temperature: 0.5, top_p: 0.9, top_k: 40,
                repeat_penalty: 1.1, max_tokens: 1024,
                n_ctx: 2048, n_gpu_layers: -1, seed: -1,
            },
            openrouter: kcm.draft.openrouter,
            behaviour: { notify_on_moderate: false, vram_threshold_mb: 500 },
        };
        kcm.dirty = true;
    }

}
