import QtQuick
import QtQuick.Layouts
import org.kde.plasma.components 3.0 as PlasmaComponents
import org.kde.kirigami as Kirigami

RowLayout {
    id: inputRoot

    signal submit(string text)

    spacing: Kirigami.Units.smallSpacing

    property alias placeholderText: inputField.placeholderText

    PlasmaComponents.TextField {
        id: inputField
        Layout.fillWidth: true
        Layout.margins: Kirigami.Units.smallSpacing
        placeholderText: "Ask anything..."
        font: Kirigami.Theme.defaultFont

        // Enter sends, Shift+Enter inserts newline
        Keys.onReturnPressed: function(event) {
            if (event.modifiers & Qt.ShiftModifier) {
                // Allow default behavior (newline)
                event.accepted = false;
            } else {
                doSend();
                event.accepted = true;
            }
        }

        // Focus input when popup opens
        Component.onCompleted: forceActiveFocus()
    }

    PlasmaComponents.ToolButton {
        icon.name: "document-send"
        enabled: inputField.text.trim().length > 0 && inputRoot.enabled
        PlasmaComponents.ToolTip { text: "Send (Enter)" }
        onClicked: doSend()
    }

    function doSend() {
        var text = inputField.text.trim();
        if (text.length === 0) return;
        inputRoot.submit(text);
        inputField.text = "";
        inputField.forceActiveFocus();
    }
}
