import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import QtQuick.Window
import org.kde.kirigami as Kirigami

/*
 * AI Assistant Settings -- standalone Kirigami.ApplicationWindow.
 *
 * This was originally a KCM (KDE Configuration Module) living under
 * /usr/share/kpackage/kcms/kcm_aios/. In KF6 the "KCModule" KPackage
 * structure was removed (see aios-settings script header), so we ship as
 * a regular desktop app. The `bridge` object is a Python Bridge injected
 * as a QML context property by aios-settings (PySide6 launcher). It
 * exposes the exact same API the previous DaemonBridge.qml exposed.
 *
 * Settings are written to ~/.config/ai-daemon/settings.yaml via the
 * aios-settings-write helper, then the daemon is asked to reload. Both
 * the plasmoid and neurosh read the same config file, so any change
 * applies OS-wide.
 */
Kirigami.ApplicationWindow {
    id: root

    // i18n shim -- KLocalizedContext is not attached (PySide6 launcher
    // doesn't ship KDE i18n bindings), so we provide a fallback that
    // does plain %1/%2/%N substitution.  When/if we switch to a C++
    // KDE host, delete this function -- the real one takes over.
    function i18n(text) {
        for (var i = 1; i < arguments.length; i++)
            text = text.replace("%" + i, String(arguments[i]));
        return text;
    }

    title: i18n("AI Assistant Settings")
    minimumWidth: Kirigami.Units.gridUnit * 32
    minimumHeight: Kirigami.Units.gridUnit * 28
    width: Kirigami.Units.gridUnit * 40
    height: Kirigami.Units.gridUnit * 34

    // Shared state -----------------------------------------------------
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
    property string openRouterKey: ""
    property string lastMessage: ""
    property string cloudFeedback: ""
    property bool cloudInferenceEnabled: false
    property bool bannerAutoHidden: false

    // Bridge wiring ----------------------------------------------------
    // `bridge` is set by the Python launcher via
    // engine.rootContext().setContextProperty("bridge", Bridge()).
    Connections {
        target: bridge
        function onStatusReceived(status) { root.daemonStatus = status; }
        function onModelsReceived(list) { root.models = list; }
        function onOperationResult(op, ok, msg) {
            root.lastMessage = ok ? (op + ": ok") : (op + ": " + msg);
            if (op === "test" || op === "setkey") {
                root.cloudFeedback = ok ? (op === "test" ? "Connection successful!" : "Key saved.")
                                        : (msg || "Operation failed");
                cloudFeedbackTimer.restart();
            }
            if (op === "load" || op === "unload" || op === "reload"
                || op === "switch" || op === "autostart")
                bridge.fetchStatus();
        }
    }

    Timer {
        interval: 2000
        running: true
        repeat: true
        triggeredOnStart: true
        onTriggered: {
            bridge.fetchStatus();
            if (root.models.length === 0)
                bridge.fetchModels();
        }
    }

    Timer {
        id: bannerDismissTimer
        interval: 10000
        onTriggered: root.bannerAutoHidden = true
    }

    Timer {
        id: cloudFeedbackTimer
        interval: 8000
        onTriggered: root.cloudFeedback = ""
    }

    onDaemonStatusChanged: {
        bannerAutoHidden = false;
        if (daemonStatus.model_loaded && daemonStatus.backend !== "offline" && !daemonStatus.loading)
            bannerDismissTimer.restart();
        else
            bannerDismissTimer.stop();
    }

    Component.onCompleted: {
        bridge.fetchStatus();
        bridge.fetchModels();
    }

    // Body -------------------------------------------------------------
    pageStack.initialPage: Kirigami.Page {
        id: page
        title: i18n("AI Assistant")
        padding: Kirigami.Units.largeSpacing

        ColumnLayout {
            anchors.fill: parent
            spacing: Kirigami.Units.largeSpacing

            // Status banner -------------------------------------------
            Kirigami.InlineMessage {
                Layout.fillWidth: true
                visible: {
                    if (daemonStatus.backend === "offline") return true;
                    if (daemonStatus.loading) return true;
                    if (!daemonStatus.model_loaded) return true;
                    return !root.bannerAutoHidden;
                }
                type: daemonStatus.backend === "offline" ? Kirigami.MessageType.Warning
                      : daemonStatus.loading ? Kirigami.MessageType.Information
                      : daemonStatus.model_loaded ? Kirigami.MessageType.Positive
                      : Kirigami.MessageType.Information
                text: {
                    if (daemonStatus.backend === "offline")
                        return i18n("Daemon offline. Start with: systemctl --user start ai-daemon");
                    if (daemonStatus.loading)
                        return i18n("Loading model...");
                    if (daemonStatus.model_loaded)
                        return i18n("Model loaded: %1 (%2, RAM %3 MB)",
                                    daemonStatus.model,
                                    daemonStatus.backend.toUpperCase(),
                                    daemonStatus.ram_used_mb);
                    return i18n("Daemon running -- model not loaded (lazy-load is on). "
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
                Layout.fillHeight: true
                currentIndex: tabs.currentIndex

                // ---- 1. Model ----------------------------------------
                Kirigami.FormLayout {
                    QQC2.Label {
                        Kirigami.FormData.label: i18n("Active model:")
                        text: daemonStatus.model || i18n("(none)")
                    }
                    QQC2.Label {
                        Kirigami.FormData.label: i18n("State:")
                        text: daemonStatus.loading ? i18n("Loading...")
                              : daemonStatus.model_loaded
                                ? i18n("Loaded  -  RAM %1 MB  -  VRAM %2 / %3 MB",
                                       daemonStatus.ram_used_mb,
                                       daemonStatus.vram_used_mb,
                                       daemonStatus.vram_used_mb + daemonStatus.vram_free_mb)
                                : i18n("Not loaded")
                    }

                    QQC2.ComboBox {
                        id: modelBox
                        Kirigami.FormData.label: i18n("Select model:")
                        Layout.fillWidth: true
                        model: root.models.filter(function(m) { return m.type === "gguf"; })
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

                // ---- 2. Cloud ----------------------------------------
                Kirigami.FormLayout {
                    QQC2.CheckBox {
                        id: cloudEnabled
                        Kirigami.FormData.label: i18n("OpenRouter:")
                        text: i18n("Enable cloud inference")
                        checked: root.cloudInferenceEnabled
                        onToggled: {
                            root.draft.openrouter.enabled = checked;
                            root.cloudInferenceEnabled = checked;
                            root.dirty = true;
                        }
                    }

                    QQC2.TextField {
                        id: cloudKey
                        Kirigami.FormData.label: i18n("API key:")
                        Layout.fillWidth: true
                        echoMode: TextInput.Password
                        placeholderText: i18n("sk-or-...")
                        text: root.openRouterKey
                        onEditingFinished: root.openRouterKey = text
                    }

                    QQC2.TextField {
                        id: cloudModel
                        Kirigami.FormData.label: i18n("Model ID:")
                        Layout.fillWidth: true
                        text: root.draft.openrouter.default_model
                        placeholderText: "deepseek/deepseek-chat"
                        onEditingFinished: {
                            root.draft.openrouter.default_model = text;
                            root.dirty = true;
                        }
                    }

                    RowLayout {
                        Kirigami.FormData.label: ""
                        QQC2.Button {
                            text: i18n("Save key")
                            enabled: cloudKey.text !== ""
                            onClicked: {
                                root.openRouterKey = cloudKey.text;
                                bridge.setOpenRouterKey(cloudKey.text);
                            }
                        }
                        QQC2.Button {
                            text: i18n("Test connection")
                            enabled: cloudKey.text !== ""
                            onClicked: {
                                root.cloudFeedback = i18n("Testing...");
                                bridge.testOpenRouter(cloudKey.text, cloudModel.text);
                            }
                        }
                        QQC2.Button {
                            text: i18n("Switch to cloud")
                            enabled: root.cloudInferenceEnabled
                                     && cloudModel.text !== ""
                            onClicked: bridge.switchModel(
                                "openrouter:" + cloudModel.text)
                        }
                    }

                    RowLayout {
                        Kirigami.FormData.label: ""
                        QQC2.Button {
                            text: i18n("Apply")
                            enabled: root.dirty
                            onClicked: applySettings()
                        }
                    }

                    Kirigami.InlineMessage {
                        Layout.fillWidth: true
                        visible: root.cloudFeedback !== ""
                        type: root.cloudFeedback.indexOf("successful") >= 0
                              || root.cloudFeedback.indexOf("saved") >= 0
                              || root.cloudFeedback.indexOf("Saved") >= 0
                              ? Kirigami.MessageType.Positive
                              : root.cloudFeedback === i18n("Testing...")
                                ? Kirigami.MessageType.Information
                                : Kirigami.MessageType.Error
                        text: root.cloudFeedback
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

                // ---- 3. Inference ------------------------------------
                Kirigami.FormLayout {
                    QQC2.Slider {
                        id: tempSlider
                        Kirigami.FormData.label: i18n("Temperature:")
                        from: 0.0; to: 2.0; stepSize: 0.05
                        value: root.draft.generation.temperature
                        onMoved: {
                            root.draft.generation.temperature = Math.round(value*100)/100;
                            root.dirty = true;
                        }
                    }
                    QQC2.Label { text: tempSlider.value.toFixed(2) }

                    QQC2.Slider {
                        id: topPSlider
                        Kirigami.FormData.label: i18n("Top-p:")
                        from: 0.0; to: 1.0; stepSize: 0.01
                        value: root.draft.generation.top_p
                        onMoved: {
                            root.draft.generation.top_p = Math.round(value*100)/100;
                            root.dirty = true;
                        }
                    }
                    QQC2.Label { text: topPSlider.value.toFixed(2) }

                    QQC2.SpinBox {
                        Kirigami.FormData.label: i18n("Top-k (0 = off):")
                        from: 0; to: 200
                        value: root.draft.generation.top_k
                        onValueModified: {
                            root.draft.generation.top_k = value;
                            root.dirty = true;
                        }
                    }

                    QQC2.Slider {
                        id: repeatSlider
                        Kirigami.FormData.label: i18n("Repeat penalty:")
                        from: 1.0; to: 1.5; stepSize: 0.01
                        value: root.draft.generation.repeat_penalty
                        onMoved: {
                            root.draft.generation.repeat_penalty = Math.round(value*100)/100;
                            root.dirty = true;
                        }
                    }
                    QQC2.Label { text: repeatSlider.value.toFixed(2) }

                    QQC2.SpinBox {
                        Kirigami.FormData.label: i18n("Max output tokens:")
                        from: 64; to: 8192; stepSize: 64
                        value: root.draft.generation.max_tokens
                        onValueModified: {
                            root.draft.generation.max_tokens = value;
                            root.dirty = true;
                        }
                    }

                    QQC2.ComboBox {
                        id: ctxBox
                        Kirigami.FormData.label: i18n("Context window:")
                        model: [1024, 2048, 4096, 8192]
                        currentIndex: Math.max(0, model.indexOf(root.draft.generation.n_ctx))
                        onActivated: {
                            root.draft.generation.n_ctx = model[currentIndex];
                            root.dirty = true;
                        }
                    }

                    QQC2.SpinBox {
                        Kirigami.FormData.label: i18n("GPU layers (-1 = all):")
                        from: -1; to: 200
                        value: root.draft.generation.n_gpu_layers
                        onValueModified: {
                            root.draft.generation.n_gpu_layers = value;
                            root.dirty = true;
                        }
                    }

                    QQC2.SpinBox {
                        Kirigami.FormData.label: i18n("Seed (-1 = random):")
                        from: -1; to: 2147483647
                        value: root.draft.generation.seed
                        onValueModified: {
                            root.draft.generation.seed = value;
                            root.dirty = true;
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
                            enabled: root.dirty
                            onClicked: applySettings()
                        }
                        QQC2.Button {
                            text: i18n("Reload model")
                            enabled: daemonStatus.model_loaded && !daemonStatus.loading
                            QQC2.ToolTip.text: i18n("Save settings, then unload and reload the model to apply context window and GPU layer changes.")
                            onClicked: {
                                var modelName = daemonStatus.model || "";
                                if (modelName !== "") {
                                    bridge.applyAndReloadModel(
                                        JSON.stringify(root.draft), modelName);
                                    root.dirty = false;
                                }
                            }
                        }
                        QQC2.Button {
                            text: i18n("Reset to defaults")
                            onClicked: resetToDefaults()
                        }
                    }
                }

                // ---- 4. Behaviour ------------------------------------
                Kirigami.FormLayout {
                    QQC2.CheckBox {
                        text: i18n("Auto-start daemon on login")
                        checked: root.draft.model.auto_start
                        onToggled: {
                            root.draft.model.auto_start = checked;
                            root.dirty = true;
                            bridge.setAutoStart(checked);
                        }
                    }
                    QQC2.CheckBox {
                        text: i18n("Lazy-load model (defer until first query)")
                        checked: root.draft.model.lazy_load
                        onToggled: {
                            root.draft.model.lazy_load = checked;
                            root.dirty = true;
                        }
                    }
                    QQC2.CheckBox {
                        text: i18n("Notify on MODERATE-risk commands")
                        checked: root.draft.behaviour.notify_on_moderate
                        onToggled: {
                            root.draft.behaviour.notify_on_moderate = checked;
                            root.dirty = true;
                        }
                    }

                    QQC2.SpinBox {
                        Kirigami.FormData.label: i18n("VRAM warning threshold (MB):")
                        from: 0; to: 4096; stepSize: 50
                        value: root.draft.behaviour.vram_threshold_mb
                        onValueModified: {
                            root.draft.behaviour.vram_threshold_mb = value;
                            root.dirty = true;
                        }
                    }

                    QQC2.Button {
                        Kirigami.FormData.label: ""
                        text: i18n("Apply")
                        enabled: root.dirty
                        onClicked: applySettings()
                    }
                }

                // ---- 5. About ----------------------------------------
                Kirigami.FormLayout {
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
                        text: root.lastMessage
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }
                }
            }
        }

        // Unload confirm --------------------------------------------
        QQC2.Dialog {
            id: unloadConfirm
            title: i18n("Unload model?")
            standardButtons: QQC2.Dialog.Ok | QQC2.Dialog.Cancel
            modal: true
            anchors.centerIn: parent
            onAccepted: bridge.unloadModel()

            contentItem: QQC2.Label {
                wrapMode: Text.Wrap
                text: i18n("This frees the model from RAM and VRAM. "
                           + "The next query will reload it "
                           + "(typically 3-5 s on GPU, longer on CPU). Continue?")
            }
        }
    }

    // Helpers ----------------------------------------------------------
    function formatUptime(s) {
        if (!s) return "-";
        var h = Math.floor(s / 3600);
        var m = Math.floor((s % 3600) / 60);
        return (h > 0 ? h + "h " : "") + m + "m";
    }

    function applySettings() {
        bridge.saveSettings(JSON.stringify(root.draft));
        root.dirty = false;
    }

    function resetToDefaults() {
        root.draft = {
            model: { default_local: root.draft.model.default_local,
                     lazy_load: true, auto_start: false },
            generation: {
                temperature: 0.5, top_p: 0.9, top_k: 40,
                repeat_penalty: 1.1, max_tokens: 1024,
                n_ctx: 2048, n_gpu_layers: -1, seed: -1,
            },
            openrouter: root.draft.openrouter,
            behaviour: { notify_on_moderate: false, vram_threshold_mb: 500 },
        };
        root.dirty = true;
    }
}
