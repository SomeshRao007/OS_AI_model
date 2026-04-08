import QtQuick
import org.kde.plasma.plasma5support as P5Support

/*
 * Thin D-Bus wrapper for org.aios.Daemon.
 *
 * The KCM cannot import python-dbus directly, so every call goes through a
 * tiny shell-out to a helper script (or the existing aios-panel-bridge for
 * the subcommands it already exposes). Settings writes go through a
 * purpose-built helper: `aios-settings-write`.
 *
 * One short-lived process per call — same pattern as the Plasmoid.
 */
QtObject {
    id: bridge

    signal statusReceived(var status)
    signal modelsReceived(var models)
    signal operationResult(string op, bool ok, string message)

    property var _runner: P5Support.DataSource {
        engine: "executable"
        connectedSources: []
        onNewData: function(source, data) {
            var stdout = (data["stdout"] || "").trim();
            var tag = _tagFor(source);
            disconnectSource(source);
            _dispatch(tag, stdout);
        }
    }
    property var _pending: ({})
    property int _nextId: 0

    function _tagFor(source) {
        for (var k in _pending) {
            if (_pending[k] === source)
                return k;
        }
        return "";
    }

    function _dispatch(tag, stdout) {
        if (!tag) return;
        var op = tag.split(":")[0];
        delete _pending[tag];

        if (op === "status") {
            try { statusReceived(JSON.parse(stdout)); }
            catch (e) { statusReceived({backend: "offline"}); }
        } else if (op === "models") {
            try { modelsReceived(JSON.parse(stdout)); }
            catch (e) { modelsReceived([]); }
        } else if (op === "load" || op === "unload" || op === "reload"
                   || op === "setkey" || op === "test" || op === "switch"
                   || op === "autostart") {
            var ok = stdout.indexOf("ok") === 0;
            var msg = ok ? "" : stdout.replace(/^error:\s*/, "");
            operationResult(op, ok, msg);
        }
    }

    function _run(op, cmd) {
        var tag = op + ":" + (_nextId++);
        _pending[tag] = cmd;
        _runner.connectSource(cmd);
    }

    // Public API
    function fetchStatus()   { _run("status", "aios-panel-bridge status"); }
    function fetchModels()   { _run("models", "aios-panel-bridge models"); }
    function loadModel(name) { _run("load",   "aios-settings-write load '" + _esc(name) + "'"); }
    function unloadModel()   { _run("unload", "aios-settings-write unload"); }
    function reloadSettings(){ _run("reload", "aios-settings-write reload"); }
    function setOpenRouterKey(key) {
        _run("setkey", "aios-settings-write set-key '" + _esc(key) + "'");
    }
    function testOpenRouter(key, modelId) {
        _run("test", "aios-settings-write test-openrouter '" + _esc(key)
             + "' '" + _esc(modelId) + "'");
    }
    function switchModel(name) {
        _run("switch", "aios-panel-bridge switch '" + _esc(name) + "'");
    }
    function setAutoStart(enabled) {
        _run("autostart", "aios-settings-write autostart "
             + (enabled ? "enable" : "disable"));
    }

    // Save a JSON settings payload, then trigger a daemon reload. The two
    // operations are chained in one shell so the KCM only has to wait on a
    // single result.
    function saveSettings(jsonPayload) {
        var esc = _esc(jsonPayload);
        _run("reload",
             "aios-settings-write save '" + esc + "' && "
             + "aios-settings-write reload");
    }

    function _esc(s) { return String(s).replace(/'/g, "'\\''"); }
}
