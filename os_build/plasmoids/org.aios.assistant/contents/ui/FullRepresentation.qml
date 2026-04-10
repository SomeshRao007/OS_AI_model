import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.plasma.components 3.0 as PlasmaComponents
import org.kde.plasma.extras as PlasmaExtras
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasma5support as P5Support

ColumnLayout {
    id: fullRoot

    readonly property var appState: root.daemonState

    Layout.preferredWidth: Kirigami.Units.gridUnit * 24
    Layout.preferredHeight: Kirigami.Units.gridUnit * 32
    Layout.minimumWidth: Kirigami.Units.gridUnit * 18
    Layout.minimumHeight: Kirigami.Units.gridUnit * 20
    spacing: 0

    // ── Header bar ────────────────────────────────────────────────────────
    Rectangle {
        Layout.fillWidth: true
        Layout.preferredHeight: headerCol.implicitHeight + Kirigami.Units.smallSpacing * 2
        color: Kirigami.Theme.backgroundColor
        Kirigami.Separator { anchors.bottom: parent.bottom; anchors.left: parent.left; anchors.right: parent.right }

        ColumnLayout {
            id: headerCol
            anchors.fill: parent
            anchors.margins: Kirigami.Units.smallSpacing
            spacing: 2

            RowLayout {
                spacing: Kirigami.Units.smallSpacing

                Kirigami.Icon {
                    source: Qt.resolvedUrl("../icons/nbs-assistant.svg")
                    Layout.preferredWidth: Kirigami.Units.iconSizes.small
                    Layout.preferredHeight: Kirigami.Units.iconSizes.small
                }

                PlasmaExtras.Heading {
                    text: "NBS Assistant"
                    level: 4
                    Layout.fillWidth: true
                }

                // Backend badge
                Rectangle {
                    visible: appState.backend !== "offline"
                    Layout.preferredHeight: badgeText.implicitHeight + Kirigami.Units.smallSpacing
                    Layout.preferredWidth: badgeText.implicitWidth + Kirigami.Units.largeSpacing
                    radius: height / 2
                    color: {
                        switch (appState.backend) {
                            case "gpu": return "#27ae60";
                            case "cpu": return "#f39c12";
                            case "openrouter": return "#3498db";
                            default: return "#95a5a6";
                        }
                    }
                    PlasmaComponents.Label {
                        id: badgeText
                        anchors.centerIn: parent
                        text: appState.backend.toUpperCase()
                        font.pixelSize: Kirigami.Theme.smallFont.pixelSize
                        color: "white"
                    }
                }

                // Clear button
                PlasmaComponents.ToolButton {
                    icon.name: "edit-clear-history"
                    PlasmaComponents.ToolTip { text: "Clear chat" }
                    onClicked: chatModel.clear()
                }

                // Settings button -- launches the standalone settings app
                // (see os_build/settings/). Shells out via the executable
                // DataEngine because the plasmoid has no direct "open URL"
                // primitive and the app runs as a normal detached process.
                PlasmaComponents.ToolButton {
                    icon.name: "configure"
                    PlasmaComponents.ToolTip { text: "Settings" }
                    onClicked: settingsLauncher.connectSource("aios-settings &")
                }
            }

            // One-shot DataSource used only to fire-and-forget the settings
            // app launcher. `aios-settings &` detaches immediately so the
            // finished signal arrives right away and we disconnect.
            P5Support.DataSource {
                id: settingsLauncher
                engine: "executable"
                connectedSources: []
                onNewData: function(source, data) {
                    disconnectSource(source);
                }
            }

            // Model + context info line
            PlasmaComponents.Label {
                visible: appState.backend !== "offline"
                Layout.fillWidth: true
                text: {
                    var parts = [appState.model];
                    if (appState.lastCompletionTokens > 0) {
                        var total = appState.lastPromptTokens + appState.lastCompletionTokens;
                        parts.push(total + " tokens");
                    }
                    if (appState.lastTokPerS > 0)
                        parts.push(appState.lastTokPerS.toFixed(1) + " tok/s");
                    return parts.join(" | ");
                }
                font: Kirigami.Theme.smallFont
                color: Kirigami.Theme.disabledTextColor
                elide: Text.ElideRight
            }
        }
    }

    // ── Chat area ─────────────────────────────────────────────────────────
    ListView {
        id: chatView
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        spacing: Kirigami.Units.smallSpacing
        topMargin: Kirigami.Units.smallSpacing
        bottomMargin: Kirigami.Units.smallSpacing

        model: ListModel { id: chatModel }
        delegate: ChatMessage {
            width: chatView.width - Kirigami.Units.largeSpacing
            anchors.horizontalCenter: parent ? parent.horizontalCenter : undefined
        }

        onCountChanged: {
            Qt.callLater(function() { chatView.positionViewAtEnd(); });
        }

        // Empty state — only when no messages AND not thinking
        PlasmaExtras.PlaceholderMessage {
            anchors.centerIn: parent
            visible: chatModel.count === 0 && !appState.thinking
            iconName: "dialog-messages"
            text: "Ask me anything"
            explanation: "I can help with system admin, files, networking, packages, and more."
        }
    }

    // ── Thinking indicator ────────────────────────────────────────────────
    RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: Kirigami.Units.largeSpacing
        Layout.bottomMargin: Kirigami.Units.smallSpacing
        visible: appState.thinking
        spacing: Kirigami.Units.smallSpacing

        Repeater {
            model: 3
            Rectangle {
                width: Kirigami.Units.smallSpacing * 2
                height: width
                radius: width / 2
                color: Kirigami.Theme.disabledTextColor
                SequentialAnimation on opacity {
                    running: appState.thinking
                    loops: Animation.Infinite
                    PauseAnimation { duration: index * 200 }
                    NumberAnimation { to: 1.0; duration: 300 }
                    NumberAnimation { to: 0.3; duration: 300 }
                    PauseAnimation { duration: (2 - index) * 200 }
                }
            }
        }

        PlasmaComponents.Label {
            // When the model isn't resident yet, tell the user what's
            // actually happening — "Thinking..." would be a lie during a
            // 3-20s cold load.
            text: (appState.loading || !appState.modelLoaded)
                  ? "Loading model..."
                  : "Thinking..."
            font: Kirigami.Theme.smallFont
            color: Kirigami.Theme.disabledTextColor
        }
    }

    // ── Separator ─────────────────────────────────────────────────────────
    Kirigami.Separator { Layout.fillWidth: true }

    // ── Input bar ─────────────────────────────────────────────────────────
    InputBar {
        Layout.fillWidth: true
        enabled: !appState.thinking
        onSubmit: function(text) {
            sendQuery(text);
        }
    }

    // ── Query execution ───────────────────────────────────────────────────
    P5Support.DataSource {
        id: querySource
        engine: "executable"
        connectedSources: []

        onNewData: function(source, data) {
            var stdout = data["stdout"] || "";
            handleQueryResponse(stdout);
            disconnectSource(source);
        }
    }

    // Safety timeout — if query takes >600s, stop thinking indicator.
    // Matches the bridge's cold-load timeout (lazy-load first call can
    // need both a model load and CPU inference).
    Timer {
        id: thinkingTimeout
        interval: 600000
        onTriggered: {
            if (appState.thinking) {
                appState.thinking = false;
                chatModel.append({
                    role: "system",
                    text: "Request timed out. The daemon may be busy with CPU inference.",
                    statsJson: "",
                });
            }
        }
    }

    function sendQuery(question) {
        chatModel.append({ role: "user", text: question, statsJson: "" });
        appState.thinking = true;
        thinkingTimeout.start();

        var escaped = question.replace(/'/g, "'\\''");
        querySource.connectSource("aios-panel-bridge query '" + escaped + "'");
    }

    function handleQueryResponse(raw) {
        appState.thinking = false;
        thinkingTimeout.stop();

        var parts = raw.split("---STATS---");
        var responseText = (parts[0] || "").trim();
        var statsJson = (parts[1] || "{}").trim();

        try {
            var stats = JSON.parse(statsJson);
            if (stats.prompt_tokens)
                appState.lastPromptTokens = stats.prompt_tokens;
            if (stats.completion_tokens)
                appState.lastCompletionTokens = stats.completion_tokens;
            if (stats.elapsed_ms)
                appState.lastElapsedMs = stats.elapsed_ms;
            if (stats.tok_per_s)
                appState.lastTokPerS = stats.tok_per_s;
        } catch (e) {}

        if (responseText) {
            chatModel.append({ role: "ai", text: responseText, statsJson: statsJson });
        } else {
            chatModel.append({
                role: "system",
                text: "No response received. Is the daemon running?",
                statsJson: "",
            });
        }
    }
}
