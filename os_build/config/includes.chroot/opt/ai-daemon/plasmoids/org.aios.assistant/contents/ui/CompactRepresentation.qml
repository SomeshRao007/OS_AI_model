import QtQuick
import QtQuick.Layouts
import org.kde.plasma.core as PlasmaCore
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasma5support as P5Support

MouseArea {
    id: compactRoot

    readonly property var state: root.daemonState

    // Status color mapping
    readonly property color statusColor: {
        switch (state.backend) {
            case "gpu":       return "#27ae60";  // green
            case "cpu":       return "#f39c12";  // yellow/amber
            case "openrouter": return "#3498db"; // blue
            default:          return "#e74c3c";  // red (offline)
        }
    }

    onClicked: root.expanded = !root.expanded
    hoverEnabled: true

    // Panel icon
    Kirigami.Icon {
        id: mainIcon
        anchors.fill: parent
        source: "system-help"
        active: compactRoot.containsMouse
    }

    // Status dot (bottom-right corner)
    Rectangle {
        width: Kirigami.Units.smallSpacing * 2.5
        height: width
        radius: width / 2
        color: statusColor
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 1

        // Pulse animation when thinking
        SequentialAnimation on opacity {
            running: state.thinking
            loops: Animation.Infinite
            NumberAnimation { to: 0.3; duration: 600; easing.type: Easing.InOutQuad }
            NumberAnimation { to: 1.0; duration: 600; easing.type: Easing.InOutQuad }
        }
    }

    // Poll daemon status every 5 seconds
    P5Support.DataSource {
        id: statusPoller
        engine: "executable"
        connectedSources: []

        onNewData: function(source, data) {
            var stdout = data["stdout"] || "";
            state.updateStatus(stdout.trim());
            disconnectSource(source);
        }
    }

    Timer {
        interval: 5000
        running: true
        repeat: true
        triggeredOnStart: true
        onTriggered: {
            statusPoller.connectSource("aios-panel-bridge status");
        }
    }
}
