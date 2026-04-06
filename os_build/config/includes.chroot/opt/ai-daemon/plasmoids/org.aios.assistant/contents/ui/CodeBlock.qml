import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.plasma.components 3.0 as PlasmaComponents
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasma5support as P5Support

Rectangle {
    id: codeRoot

    required property string language
    required property string code

    implicitHeight: codeLayout.implicitHeight + Kirigami.Units.largeSpacing
    radius: Kirigami.Units.smallSpacing
    color: "#1e1e2e"  // Dark background — code convention, always dark

    ColumnLayout {
        id: codeLayout
        anchors.fill: parent
        anchors.margins: Kirigami.Units.smallSpacing
        spacing: Kirigami.Units.smallSpacing

        // Header: language label + action buttons
        RowLayout {
            Layout.fillWidth: true
            spacing: Kirigami.Units.smallSpacing

            PlasmaComponents.Label {
                text: language
                font.pixelSize: Kirigami.Theme.smallFont.pixelSize
                color: "#89b4fa"  // Catppuccin blue
                opacity: 0.8
            }

            Item { Layout.fillWidth: true }

            // Copy button
            PlasmaComponents.ToolButton {
                icon.name: copyFeedback.running ? "dialog-ok" : "edit-copy"
                icon.color: "#cdd6f4"
                implicitWidth: Kirigami.Units.iconSizes.small + Kirigami.Units.smallSpacing
                implicitHeight: implicitWidth
                PlasmaComponents.ToolTip { text: "Copy to clipboard" }
                onClicked: {
                    textEdit.selectAll();
                    textEdit.copy();
                    textEdit.deselect();
                    copyFeedback.start();
                }

                Timer {
                    id: copyFeedback
                    interval: 1500
                }
            }

            // Run button (only for bash/sh)
            PlasmaComponents.ToolButton {
                visible: language === "bash" || language === "sh" || language === ""
                icon.name: runSource.running ? "media-playback-stop" : "media-playback-start"
                icon.color: "#a6e3a1"  // Catppuccin green
                implicitWidth: Kirigami.Units.iconSizes.small + Kirigami.Units.smallSpacing
                implicitHeight: implicitWidth
                PlasmaComponents.ToolTip { text: "Run command" }
                onClicked: {
                    if (!runSource.running) {
                        outputText.text = "";
                        outputArea.visible = true;
                        var escaped = code.replace(/'/g, "'\\''");
                        runSource.connectSource("aios-panel-bridge exec '" + escaped + "'");
                    }
                }

                property bool running: false
            }
        }

        // Code text (selectable + copyable via hidden TextEdit)
        TextEdit {
            id: textEdit
            Layout.fillWidth: true
            text: code
            readOnly: true
            selectByMouse: true
            wrapMode: Text.Wrap
            font.family: "monospace"
            font.pixelSize: Kirigami.Theme.defaultFont.pixelSize * 0.9
            color: "#cdd6f4"  // Catppuccin text
            selectionColor: "#585b70"
        }

        // Command output area (shown after Run)
        Rectangle {
            id: outputArea
            visible: false
            Layout.fillWidth: true
            implicitHeight: outputText.implicitHeight + Kirigami.Units.smallSpacing * 2
            radius: Kirigami.Units.smallSpacing / 2
            color: "#11111b"  // Darker than code bg
            border.color: "#313244"
            border.width: 1

            PlasmaComponents.Label {
                id: outputText
                anchors.fill: parent
                anchors.margins: Kirigami.Units.smallSpacing
                text: "Running..."
                font.family: "monospace"
                font.pixelSize: Kirigami.Theme.smallFont.pixelSize
                color: "#a6adc8"
                wrapMode: Text.Wrap
                textFormat: Text.PlainText
            }
        }
    }

    // Command runner
    P5Support.DataSource {
        id: runSource
        engine: "executable"
        connectedSources: []

        property bool running: connectedSources.length > 0

        onNewData: function(source, data) {
            var stdout = data["stdout"] || "";
            var stderr = data["stderr"] || "";
            outputText.text = (stdout + stderr).trim() || "(no output)";
            disconnectSource(source);
        }
    }

    // Fade-in
    opacity: 0
    Component.onCompleted: opacity = 1
    Behavior on opacity { NumberAnimation { duration: 200 } }
}
