import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.plasma.components 3.0 as PlasmaComponents
import org.kde.kirigami as Kirigami

Item {
    id: messageRoot

    required property string role
    required property string text
    property string statsJson: ""

    readonly property bool isUser: role === "user"
    readonly property bool isSystem: role === "system"
    readonly property bool isAi: role === "ai"

    readonly property string footerText: {
        if (!isAi || !statsJson)
            return "";
        try {
            var s = JSON.parse(statsJson);
            if (s.error)
                return "";
            var bits = [];
            var promptT = s.prompt_tokens || 0;
            var compT = s.completion_tokens || 0;
            var total = promptT + compT;
            if (total > 0)
                bits.push(total + " tok (" + promptT + "+" + compT + ")");
            if (s.backend === "openrouter") {
                if (s.cost_usd !== undefined && s.cost_usd > 0)
                    bits.push("$" + s.cost_usd.toFixed(5));
                else if (s.cost_usd === 0)
                    bits.push("$0.00000");
            } else {
                if (s.tok_per_s)
                    bits.push(s.tok_per_s.toFixed(1) + " tok/s");
                else if (s.elapsed_s)
                    bits.push(s.elapsed_s.toFixed(2) + "s");
            }
            return bits.join("  ·  ");
        } catch (e) {
            return "";
        }
    }

    implicitHeight: bubbleColumn.implicitHeight + Kirigami.Units.smallSpacing

    ColumnLayout {
        id: bubbleColumn
        width: parent.width
        spacing: Kirigami.Units.smallSpacing

        Repeater {
            model: parseSegments(messageRoot.text)

            Loader {
                Layout.fillWidth: true
                Layout.alignment: isUser ? Qt.AlignRight : Qt.AlignLeft
                Layout.maximumWidth: messageRoot.width * 0.85

                sourceComponent: modelData.type === "code"
                    ? codeBlockComponent
                    : textBubbleComponent

                property var segmentData: modelData
            }
        }

        PlasmaComponents.Label {
            visible: messageRoot.footerText !== ""
            Layout.alignment: Qt.AlignLeft
            Layout.leftMargin: Kirigami.Units.smallSpacing
            text: messageRoot.footerText
            font: Kirigami.Theme.smallFont
            opacity: 0.65
            color: Kirigami.Theme.disabledTextColor
        }
    }

    Component {
        id: textBubbleComponent

        Rectangle {
            implicitHeight: contentLabel.implicitHeight + Kirigami.Units.largeSpacing
            implicitWidth: Math.min(
                contentLabel.implicitWidth + Kirigami.Units.largeSpacing * 2,
                messageRoot.width * 0.85
            )

            radius: Kirigami.Units.smallSpacing * 2
            color: {
                if (isSystem)
                    return Qt.rgba(Kirigami.Theme.negativeTextColor.r,
                                   Kirigami.Theme.negativeTextColor.g,
                                   Kirigami.Theme.negativeTextColor.b, 0.15);
                if (isUser)
                    return Qt.rgba(Kirigami.Theme.highlightColor.r,
                                   Kirigami.Theme.highlightColor.g,
                                   Kirigami.Theme.highlightColor.b, 0.2);
                return Qt.rgba(Kirigami.Theme.textColor.r,
                               Kirigami.Theme.textColor.g,
                               Kirigami.Theme.textColor.b, 0.06);
            }

            PlasmaComponents.Label {
                id: contentLabel
                anchors.fill: parent
                anchors.margins: Kirigami.Units.smallSpacing * 1.5
                text: segmentData.content
                wrapMode: Text.Wrap
                textFormat: Text.PlainText
                color: Kirigami.Theme.textColor
            }

            opacity: 0
            Component.onCompleted: opacity = 1
            Behavior on opacity { NumberAnimation { duration: 200 } }
        }
    }

    Component {
        id: codeBlockComponent

        CodeBlock {
            language: segmentData.language || "bash"
            code: segmentData.content
        }
    }

    // Parse text into segments: [{type: "text"/"code", content, language}]
    function parseSegments(rawText) {
        var segments = [];
        // Match fenced code blocks: ```lang\ncode\n```
        var codePattern = /```(\w*)\n([\s\S]*?)```/g;
        var lastIndex = 0;
        var match;

        while ((match = codePattern.exec(rawText)) !== null) {
            if (match.index > lastIndex) {
                var textBefore = rawText.slice(lastIndex, match.index).trim();
                if (textBefore)
                    segments.push({ type: "text", content: textBefore });
            }
            segments.push({
                type: "code",
                language: match[1] || "bash",
                content: match[2].trim(),
            });
            lastIndex = codePattern.lastIndex;
        }

        if (lastIndex < rawText.length) {
            var remaining = rawText.slice(lastIndex).trim();
            if (remaining)
                segments.push({ type: "text", content: remaining });
        }

        if (segments.length === 0 && rawText.trim())
            segments.push({ type: "text", content: rawText.trim() });

        return segments;
    }
}
