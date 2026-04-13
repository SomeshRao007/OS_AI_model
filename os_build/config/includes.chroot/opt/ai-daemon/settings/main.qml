import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import QtQuick.Window
import org.kde.kirigami as Kirigami

/*
 * NBS Assistant Settings -- standalone Kirigami.ApplicationWindow.
 *
 * The `bridge` object is a Python Bridge injected as a QML context
 * property by aios-settings (PySide6 launcher). It exposes slots for
 * daemon control, profile management, and settings persistence.
 *
 * Settings are written to ~/.config/ai-daemon/settings.yaml via the
 * aios-settings-write helper, then the daemon is asked to reload. Both
 * the plasmoid and neurosh read the same config file, so any change
 * applies OS-wide.
 *
 * OpenRouter API keys are stored exclusively in KWallet (no plaintext
 * files). The daemon fetches keys from KWallet at runtime.
 */
Kirigami.ApplicationWindow {
    id: root

    function i18n(text) {
        for (var i = 1; i < arguments.length; i++)
            text = text.replace("%" + i, String(arguments[i]));
        return text;
    }

    title: i18n("NBS Assistant Settings")
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
    property var profiles: []
    property var draft: ({
        active: { backend: "local", local_model: "", openrouter_profile: "" },
        model: { default_local: "", lazy_load: true, auto_start: false },
        generation: {
            temperature: 0.5, top_p: 0.9, top_k: 40,
            repeat_penalty: 1.1, max_tokens: 1024,
            n_ctx: 2048, n_gpu_layers: -1, seed: -1,
        },
        openrouter: { enabled: false },
        behaviour: { notify_on_moderate: false, vram_threshold_mb: 500 },
    })
    property bool dirty: false
    property string lastMessage: ""
    property string cloudFeedback: ""
    property bool cloudInferenceEnabled: false
    property bool bannerAutoHidden: false
    readonly property bool isOpenRouter: daemonStatus.backend === "openrouter"

    // Reusable right-click context menu for text fields
    component FieldContextMenu: QQC2.Menu {
        id: _ctxMenu
        property var field
        QQC2.MenuItem {
            text: i18n("Cut")
            enabled: _ctxMenu.field && _ctxMenu.field.selectedText.length > 0
                     && !_ctxMenu.field.readOnly
            onTriggered: _ctxMenu.field.cut()
        }
        QQC2.MenuItem {
            text: i18n("Copy")
            enabled: _ctxMenu.field && _ctxMenu.field.selectedText.length > 0
            onTriggered: _ctxMenu.field.copy()
        }
        QQC2.MenuItem {
            text: i18n("Paste")
            enabled: _ctxMenu.field && _ctxMenu.field.canPaste
                     && !_ctxMenu.field.readOnly
            onTriggered: _ctxMenu.field.paste()
        }
        QQC2.MenuItem {
            text: i18n("Select All")
            onTriggered: _ctxMenu.field.selectAll()
        }
    }

    // Current profile state (derived from profiles list)
    property string currentProfileName: ""
    property string currentProfileModel: ""
    property string currentProfileMaskedKey: ""

    // Bridge wiring ----------------------------------------------------
    Connections {
        target: bridge
        function onStatusReceived(status) { root.daemonStatus = status; }
        function onModelsReceived(list) { root.models = list; }
        function onSettingsReceived(settings) {
            if (!settings || Object.keys(settings).length === 0) return;
            root.draft = settings;
            root.cloudInferenceEnabled = !!(settings.openrouter && settings.openrouter.enabled);
            root.dirty = false;
        }
        function onProfilesReceived(list) {
            root.profiles = list || [];
            // Update current profile display state
            var found = false;
            for (var i = 0; i < root.profiles.length; i++) {
                if (root.profiles[i].is_current) {
                    root.currentProfileName = root.profiles[i].name;
                    root.currentProfileModel = root.profiles[i].last_model || "";
                    root.currentProfileMaskedKey = root.profiles[i].masked_key || "";
                    found = true;
                    break;
                }
            }
            if (!found) {
                root.currentProfileName = "";
                root.currentProfileModel = "";
                root.currentProfileMaskedKey = "";
            }
        }
        function onKeyRevealed(profileName, plaintextKey) {
            if (profileName === root.currentProfileName)
                cloudKeyField.text = plaintextKey;
        }
        function onOperationResult(op, ok, msg) {
            root.lastMessage = ok ? (op + ": ok") : (op + ": " + msg);
            if (op === "test") {
                root.cloudFeedback = ok ? "Connection successful!"
                                        : (msg || "Test failed");
                cloudFeedbackTimer.restart();
            }
            if (op === "upsert" || op === "delete" || op === "setcurrent") {
                bridge.fetchProfiles();
                if (op === "upsert")
                    root.cloudFeedback = ok ? "Profile saved." : (msg || "Save failed");
                if (op === "delete")
                    root.cloudFeedback = ok ? "Profile deleted." : (msg || "Delete failed");
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
        bridge.fetchSettings();
        bridge.fetchProfiles();
    }

    // Body -------------------------------------------------------------
    pageStack.initialPage: Kirigami.Page {
        id: page
        title: i18n("NBS Assistant")
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
                    if (daemonStatus.model_loaded) {
                        if (root.isOpenRouter)
                            return i18n("Cloud inference active: %1", daemonStatus.model);
                        return i18n("Model loaded: %1 (%2, RAM %3 MB)",
                                    daemonStatus.model,
                                    daemonStatus.backend.toUpperCase(),
                                    daemonStatus.ram_used_mb);
                    }
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
                                ? (root.isOpenRouter
                                   ? i18n("Cloud — %1", daemonStatus.model)
                                   : i18n("Loaded  -  RAM %1 MB  -  VRAM %2 / %3 MB",
                                          daemonStatus.ram_used_mb,
                                          daemonStatus.vram_used_mb,
                                          daemonStatus.vram_used_mb + daemonStatus.vram_free_mb))
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
                                     && !root.isOpenRouter
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

                    // -- Profile selector --
                    RowLayout {
                        Kirigami.FormData.label: i18n("Profile:")
                        spacing: Kirigami.Units.smallSpacing

                        QQC2.ComboBox {
                            id: profileBox
                            Layout.fillWidth: true
                            model: root.profiles.map(function(p) { return p.name; })
                            currentIndex: {
                                var names = root.profiles.map(function(p) { return p.name; });
                                return Math.max(0, names.indexOf(root.currentProfileName));
                            }
                            onActivated: {
                                var name = profileBox.currentText;
                                if (name && name !== root.currentProfileName) {
                                    bridge.setCurrentProfile(name);
                                }
                            }
                        }
                        QQC2.Button {
                            text: i18n("+ Add")
                            onClicked: addProfileDialog.open()
                        }
                        QQC2.Button {
                            text: i18n("Delete")
                            enabled: root.currentProfileName !== ""
                            onClicked: {
                                if (root.profiles.length <= 1 && root.isOpenRouter) {
                                    deleteLastProfileConfirm.open();
                                } else {
                                    deleteProfileConfirm.open();
                                }
                            }
                        }
                    }

                    // -- Key display with eye toggle --
                    RowLayout {
                        Kirigami.FormData.label: i18n("API key:")
                        spacing: Kirigami.Units.smallSpacing

                        QQC2.TextField {
                            id: cloudKeyField
                            Layout.fillWidth: true
                            echoMode: keyRevealed ? TextInput.Normal : TextInput.Password
                            placeholderText: root.currentProfileName
                                ? root.currentProfileMaskedKey || i18n("(no key)")
                                : i18n("Select or add a profile first")
                            text: ""
                            readOnly: true

                            property bool keyRevealed: false

                            TapHandler {
                                acceptedButtons: Qt.RightButton
                                onTapped: cloudKeyMenu.popup()
                            }
                            FieldContextMenu { id: cloudKeyMenu; field: cloudKeyField }
                        }
                        QQC2.ToolButton {
                            icon.name: cloudKeyField.keyRevealed
                                       ? "password-show-off" : "password-show-on"
                            QQC2.ToolTip.text: cloudKeyField.keyRevealed
                                               ? i18n("Hide key") : i18n("Reveal key")
                            QQC2.ToolTip.visible: hovered
                            enabled: root.currentProfileName !== ""
                            onClicked: {
                                if (cloudKeyField.keyRevealed) {
                                    cloudKeyField.text = "";
                                    cloudKeyField.keyRevealed = false;
                                } else {
                                    bridge.revealProfile(root.currentProfileName);
                                    cloudKeyField.keyRevealed = true;
                                }
                            }
                        }
                    }

                    QQC2.Label {
                        Kirigami.FormData.label: i18n("Model:")
                        text: root.currentProfileModel || i18n("(none)")
                    }

                    RowLayout {
                        Kirigami.FormData.label: ""
                        spacing: Kirigami.Units.smallSpacing

                        QQC2.Button {
                            text: i18n("Test connection")
                            enabled: root.currentProfileName !== ""
                                     && root.currentProfileModel !== ""
                            onClicked: {
                                root.cloudFeedback = i18n("Testing...");
                                bridge.testOpenRouter(root.currentProfileName);
                            }
                        }
                        QQC2.Button {
                            text: i18n("Switch to cloud")
                            enabled: root.cloudInferenceEnabled
                                     && root.currentProfileName !== ""
                                     && root.currentProfileModel !== ""
                            onClicked: bridge.switchModel(
                                "openrouter:" + root.currentProfileModel)
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
                              || root.cloudFeedback.indexOf("deleted") >= 0
                              ? Kirigami.MessageType.Positive
                              : root.cloudFeedback === i18n("Testing...")
                                ? Kirigami.MessageType.Information
                                : Kirigami.MessageType.Error
                        text: root.cloudFeedback
                    }

                    QQC2.Label {
                        Kirigami.FormData.label: ""
                        text: i18n("API keys are stored encrypted in KWallet. "
                                   + "Plaintext keys never touch the filesystem.")
                        font: Kirigami.Theme.smallFont
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                        color: Kirigami.Theme.disabledTextColor
                    }
                }

                // ---- 3. Inference ------------------------------------
                Kirigami.FormLayout {
                    // Temperature / top_p / max_tokens work on both backends
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

                    // Local-only params — disabled when OpenRouter is active
                    QQC2.SpinBox {
                        Kirigami.FormData.label: i18n("Top-k (0 = off):")
                        from: 0; to: 200
                        value: root.draft.generation.top_k
                        enabled: !root.isOpenRouter
                        QQC2.ToolTip.text: i18n("Local-only — has no effect on OpenRouter")
                        QQC2.ToolTip.visible: hovered && root.isOpenRouter
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
                        enabled: !root.isOpenRouter
                        onMoved: {
                            root.draft.generation.repeat_penalty = Math.round(value*100)/100;
                            root.dirty = true;
                        }
                    }
                    RowLayout {
                        QQC2.Label { text: repeatSlider.value.toFixed(2) }
                        QQC2.Label {
                            visible: root.isOpenRouter
                            text: i18n("(local-only)")
                            color: Kirigami.Theme.disabledTextColor
                            font: Kirigami.Theme.smallFont
                        }
                    }

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
                        enabled: !root.isOpenRouter
                        QQC2.ToolTip.text: i18n("Local-only — has no effect on OpenRouter")
                        QQC2.ToolTip.visible: hovered && root.isOpenRouter
                        onActivated: {
                            root.draft.generation.n_ctx = model[currentIndex];
                            root.dirty = true;
                        }
                    }

                    QQC2.SpinBox {
                        Kirigami.FormData.label: i18n("GPU layers (-1 = all):")
                        from: -1; to: 200
                        value: root.draft.generation.n_gpu_layers
                        enabled: !root.isOpenRouter
                        QQC2.ToolTip.text: i18n("Local-only — has no effect on OpenRouter")
                        QQC2.ToolTip.visible: hovered && root.isOpenRouter
                        onValueModified: {
                            root.draft.generation.n_gpu_layers = value;
                            root.dirty = true;
                        }
                    }

                    QQC2.SpinBox {
                        Kirigami.FormData.label: i18n("Seed (-1 = random):")
                        from: -1; to: 2147483647
                        value: root.draft.generation.seed
                        enabled: !root.isOpenRouter
                        QQC2.ToolTip.text: i18n("Local-only — has no effect on OpenRouter")
                        QQC2.ToolTip.visible: hovered && root.isOpenRouter
                        onValueModified: {
                            root.draft.generation.seed = value;
                            root.dirty = true;
                        }
                    }

                    Kirigami.InlineMessage {
                        Layout.fillWidth: true
                        visible: !root.isOpenRouter
                        type: Kirigami.MessageType.Information
                        text: i18n("Context window and GPU layers are reload-required. "
                                   + "Other parameters apply on the next query.")
                    }

                    Kirigami.InlineMessage {
                        Layout.fillWidth: true
                        visible: root.isOpenRouter
                        type: Kirigami.MessageType.Information
                        text: i18n("Cloud mode active. Only temperature, top-p, and "
                                   + "max tokens affect OpenRouter queries. "
                                   + "Other parameters are local-only.")
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
                            visible: !root.isOpenRouter
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

        // Add profile dialog ----------------------------------------
        QQC2.Dialog {
            id: addProfileDialog
            title: i18n("Add OpenRouter Profile")
            standardButtons: QQC2.Dialog.Ok | QQC2.Dialog.Cancel
            modal: true
            anchors.centerIn: parent
            width: Kirigami.Units.gridUnit * 24

            onAccepted: {
                if (newProfileName.text && newProfileKey.text) {
                    bridge.upsertProfile(
                        newProfileName.text,
                        newProfileKey.text,
                        newProfileModel.text
                    );
                    newProfileName.text = "";
                    newProfileKey.text = "";
                    newProfileModel.text = "";
                }
            }
            onRejected: {
                newProfileName.text = "";
                newProfileKey.text = "";
                newProfileModel.text = "";
            }

            contentItem: ColumnLayout {
                spacing: Kirigami.Units.smallSpacing

                QQC2.Label { text: i18n("Profile name:") }
                QQC2.TextField {
                    id: newProfileName
                    Layout.fillWidth: true
                    placeholderText: i18n("e.g. personal, work")
                    TapHandler {
                        acceptedButtons: Qt.RightButton
                        onTapped: nameMenu.popup()
                    }
                    FieldContextMenu { id: nameMenu; field: newProfileName }
                }
                QQC2.Label { text: i18n("API key:") }
                QQC2.TextField {
                    id: newProfileKey
                    Layout.fillWidth: true
                    echoMode: TextInput.Password
                    placeholderText: i18n("sk-or-...")
                    TapHandler {
                        acceptedButtons: Qt.RightButton
                        onTapped: keyMenu.popup()
                    }
                    FieldContextMenu { id: keyMenu; field: newProfileKey }
                }
                QQC2.Label { text: i18n("Model ID:") }
                QQC2.TextField {
                    id: newProfileModel
                    Layout.fillWidth: true
                    placeholderText: i18n("e.g. anthropic/claude-sonnet-4-6")
                    TapHandler {
                        acceptedButtons: Qt.RightButton
                        onTapped: modelMenu.popup()
                    }
                    FieldContextMenu { id: modelMenu; field: newProfileModel }
                }
                QQC2.Label {
                    text: i18n("After saving, select the profile and use "
                               + "Test Connection to verify.")
                    font: Kirigami.Theme.smallFont
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    color: Kirigami.Theme.disabledTextColor
                }
            }
        }

        // Delete profile confirm ------------------------------------
        QQC2.Dialog {
            id: deleteProfileConfirm
            title: i18n("Delete profile?")
            standardButtons: QQC2.Dialog.Ok | QQC2.Dialog.Cancel
            modal: true
            anchors.centerIn: parent
            onAccepted: bridge.deleteProfile(root.currentProfileName)

            contentItem: QQC2.Label {
                wrapMode: Text.Wrap
                text: i18n("Delete profile \"%1\"? This cannot be undone.",
                           root.currentProfileName)
            }
        }

        // Delete last profile confirm (switch to local first) -------
        QQC2.Dialog {
            id: deleteLastProfileConfirm
            title: i18n("Delete only profile?")
            standardButtons: QQC2.Dialog.Ok | QQC2.Dialog.Cancel
            modal: true
            anchors.centerIn: parent
            onAccepted: {
                // Switch to local first, then delete
                var lastLocal = root.draft.active.local_model
                    || root.draft.model.default_local || "";
                if (lastLocal)
                    bridge.loadModel(lastLocal);
                bridge.deleteProfile(root.currentProfileName);
            }

            contentItem: QQC2.Label {
                wrapMode: Text.Wrap
                text: i18n("This is your only profile. Deleting it will "
                           + "switch the daemon to the local model. Continue?")
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
            active: root.draft.active,
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
