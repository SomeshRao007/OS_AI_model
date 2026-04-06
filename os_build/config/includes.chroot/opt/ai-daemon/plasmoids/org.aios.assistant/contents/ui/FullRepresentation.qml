import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.plasma.components 3.0 as PlasmaComponents
import org.kde.plasma.extras as PlasmaExtras
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasma5support as P5Support

ColumnLayout {
    id: fullRoot

    readonly property var state: root.daemonState

    Layout.preferredWidth: Kirigami.Units.gridUnit * 24
    Layout.preferredHeight: Kirigami.Units.gridUnit * 32
    Layout.minimumWidth: Kirigami.Units.gridUnit * 18
    Layout.minimumHeight: Kirigami.Units.gridUnit * 20
    spacing: 0

    // ── Header bar ────────────────────────────────────────────────────────
    Rectangle {
        Layout.fillWidth: true
        Layout.preferredHeight: headerLayout.implicitHeight + Kirigami.Units.smallSpacing * 2
        color: Kirigami.Theme.backgroundColor
        Kirigami.Separator { anchors.bottom: parent.bottom; anchors.left: parent.left; anchors.right: parent.right }

        RowLayout {
            id: headerLayout
            anchors.fill: parent
            anchors.margins: Kirigami.Units.smallSpacing
            spacing: Kirigami.Units.smallSpacing

            Kirigami.Icon {
                source: "system-help"
                Layout.preferredWidth: Kirigami.Units.iconSizes.small
                Layout.preferredHeight: Kirigami.Units.iconSizes.small
            }

            PlasmaExtras.Heading {
                text: "AI Assistant"
                level: 4
                Layout.fillWidth: true
            }

            // Backend badge
            Rectangle {
                visible: state.backend !== "offline"
                Layout.preferredHeight: badgeText.implicitHeight + Kirigami.Units.smallSpacing
                Layout.preferredWidth: badgeText.implicitWidth + Kirigami.Units.largeSpacing
                radius: height / 2
                color: {
                    switch (state.backend) {
                        case "gpu": return "#27ae60";
                        case "cpu": return "#f39c12";
                        case "openrouter": return "#3498db";
                        default: return "#95a5a6";
                    }
                }
                PlasmaComponents.Label {
                    id: badgeText
                    anchors.centerIn: parent
                    text: state.model.length > 20
                          ? state.model.substring(0, 18) + "..."
                          : state.model
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

        // Auto-scroll to bottom on new messages
        onCountChanged: {
            Qt.callLater(function() { chatView.positionViewAtEnd(); });
        }

        // Empty state
        PlasmaExtras.PlaceholderMessage {
            anchors.centerIn: parent
            visible: chatModel.count === 0
            iconName: "system-help"
            text: "Ask me anything"
            explanation: "I can help with system admin, files, networking, packages, and more."
        }
    }

    // ── Thinking indicator ────────────────────────────────────────────────
    RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: Kirigami.Units.largeSpacing
        Layout.bottomMargin: Kirigami.Units.smallSpacing
        visible: state.thinking
        spacing: Kirigami.Units.smallSpacing

        Repeater {
            model: 3
            Rectangle {
                width: Kirigami.Units.smallSpacing * 2
                height: width
                radius: width / 2
                color: Kirigami.Theme.disabledTextColor
                SequentialAnimation on opacity {
                    running: state.thinking
                    loops: Animation.Infinite
                    PauseAnimation { duration: index * 200 }
                    NumberAnimation { to: 1.0; duration: 300 }
                    NumberAnimation { to: 0.3; duration: 300 }
                    PauseAnimation { duration: (2 - index) * 200 }
                }
            }
        }

        PlasmaComponents.Label {
            text: "Thinking..."
            font: Kirigami.Theme.smallFont
            color: Kirigami.Theme.disabledTextColor
        }
    }

    // ── Stats bar (after response) ────────────────────────────────────────
    PlasmaComponents.Label {
        Layout.fillWidth: true
        Layout.leftMargin: Kirigami.Units.largeSpacing
        Layout.rightMargin: Kirigami.Units.largeSpacing
        visible: state.lastCompletionTokens > 0 && !state.thinking
        text: {
            var parts = [];
            if (state.lastPromptTokens > 0)
                parts.push("prompt: " + state.lastPromptTokens);
            if (state.lastCompletionTokens > 0)
                parts.push("completion: " + state.lastCompletionTokens);
            if (state.lastTokPerS > 0)
                parts.push(state.lastTokPerS.toFixed(1) + " tok/s");
            return parts.join(" | ");
        }
        font: Kirigami.Theme.smallFont
        color: Kirigami.Theme.disabledTextColor
        horizontalAlignment: Text.AlignRight
    }

    // ── Separator ─────────────────────────────────────────────────────────
    Kirigami.Separator { Layout.fillWidth: true }

    // ── Input bar ─────────────────────────────────────────────────────────
    InputBar {
        Layout.fillWidth: true
        enabled: !state.thinking
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

    function sendQuery(question) {
        // Add user message
        chatModel.append({
            role: "user",
            text: question,
        });

        state.thinking = true;

        // Shell-safe: pass question via environment variable to avoid injection
        var escaped = question.replace(/'/g, "'\\''");
        querySource.connectSource("aios-panel-bridge query '" + escaped + "'");
    }

    function handleQueryResponse(raw) {
        state.thinking = false;

        // Split response and stats
        var parts = raw.split("---STATS---");
        var responseText = (parts[0] || "").trim();
        var statsJson = (parts[1] || "{}").trim();

        // Parse stats
        try {
            var stats = JSON.parse(statsJson);
            if (stats.prompt_tokens)
                state.lastPromptTokens = stats.prompt_tokens;
            if (stats.completion_tokens)
                state.lastCompletionTokens = stats.completion_tokens;
            if (stats.elapsed_ms)
                state.lastElapsedMs = stats.elapsed_ms;
            if (stats.tok_per_s)
                state.lastTokPerS = stats.tok_per_s;
        } catch (e) {}

        // Add AI response
        if (responseText) {
            chatModel.append({
                role: "ai",
                text: responseText,
            });
        } else {
            chatModel.append({
                role: "system",
                text: "No response received. Is the daemon running?",
            });
        }
    }
}
