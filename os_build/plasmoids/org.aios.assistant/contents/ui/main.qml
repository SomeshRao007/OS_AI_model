import QtQuick
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore

PlasmoidItem {
    id: root

    // Start compact (icon in panel), expand to full chat popup
    preferredRepresentation: compactRepresentation
    switchWidth: Kirigami.Units.gridUnit * 20
    switchHeight: Kirigami.Units.gridUnit * 20

    toolTipMainText: "NBS Assistant"
    toolTipSubText: internal.toolTipText

    compactRepresentation: CompactRepresentation {}
    fullRepresentation: FullRepresentation {}

    // Win+A (Meta+A) keyboard shortcut to toggle
    Plasmoid.contextualActions: [
        PlasmaCore.Action {
            text: "Toggle AI Assistant"
            shortcut: "Meta+A"
            onTriggered: root.expanded = !root.expanded
        }
    ]

    // Shared state accessible from all child components
    QtObject {
        id: internal

        // Daemon status
        property string model: "loading..."
        property string backend: "offline"
        property int vramUsed: 0
        property int vramFree: 0
        property int ramUsed: 0
        property int uptimeSeconds: 0
        // Lazy-load state (Step 9)
        property bool modelLoaded: false
        property bool loading: false
        property bool lazyLoad: true

        // Last inference stats
        property int lastPromptTokens: 0
        property int lastCompletionTokens: 0
        property int lastElapsedMs: 0
        property real lastTokPerS: 0

        // UI state
        property bool thinking: false

        property string toolTipText: {
            if (backend === "offline")
                return "Daemon offline";
            var line = "Model: " + model + "\nBackend: " + backend.toUpperCase();
            if (backend === "gpu" && vramUsed > 0)
                line += " (" + vramUsed + "/" + (vramUsed + vramFree) + " MB VRAM)";
            if (uptimeSeconds > 0)
                line += "\nUptime: " + formatUptime(uptimeSeconds);
            if (lastTokPerS > 0)
                line += "\nSpeed: " + lastTokPerS.toFixed(1) + " tok/s";
            return line;
        }

        function formatUptime(s) {
            var h = Math.floor(s / 3600);
            var m = Math.floor((s % 3600) / 60);
            if (h > 0) return h + "h " + m + "m";
            return m + "m";
        }

        function updateStatus(statusJson) {
            try {
                var s = JSON.parse(statusJson);
                model = s.model || "unknown";
                backend = s.backend || "offline";
                vramUsed = parseInt(s.vram_used_mb) || 0;
                vramFree = parseInt(s.vram_free_mb) || 0;
                ramUsed = parseInt(s.ram_used_mb) || 0;
                uptimeSeconds = parseInt(s.uptime_seconds) || 0;
                // New Step 9 fields — bridge returns real bools, so test
                // strictly against true to handle both bool and string forms.
                modelLoaded = (s.model_loaded === true || s.model_loaded === "True" || s.model_loaded === "1");
                loading = (s.loading === true || s.loading === "True" || s.loading === "1");
                lazyLoad = (s.lazy_load === true || s.lazy_load === "True" || s.lazy_load === "1");

                if (s.last_completion_tokens && s.last_elapsed_ms) {
                    lastPromptTokens = parseInt(s.last_prompt_tokens) || 0;
                    lastCompletionTokens = parseInt(s.last_completion_tokens) || 0;
                    lastElapsedMs = parseInt(s.last_elapsed_ms) || 0;
                    var secs = lastElapsedMs / 1000.0;
                    lastTokPerS = secs > 0 ? lastCompletionTokens / secs : 0;
                }
            } catch (e) {
                backend = "offline";
            }
        }
    }

    // Expose internal state to child components
    property alias daemonState: internal
}
