<?php
// SVG text-as-paths CAPTCHA (port of node svg-captcha) for ThinkPHP. Minimal stand-ins for the
// framework services it touches (session store, config facade, app()/runtime_path()); rendering is
// upstream. Controlled text replaces captchaText() after its own draw; label = drawn text.
namespace think { class Session { public array $data = []; public function set($k, $v) { $this->data[$k] = $v; } public function get($k, $d = null) { return $this->data[$k] ?? $d; } public function delete($k) { unset($this->data[$k]); } } }
namespace think\facade { class Config { public static function get($k, $d = null) { return $d; } } }
namespace {
    // ThinkPHP app() helper: the font loader resolves fonts under <root>/config/font/.
    function app() { return new class { public function getRootPath() { return $GLOBALS['BANK_CFG']['tmp'] . '/'; } public function getRuntimePath() { return $GLOBALS['BANK_CFG']['tmp'] . '/runtime/'; } }; }
    function runtime_path($p = '') { return $GLOBALS['BANK_CFG']['tmp'] . '/runtime/' . ($p ? $p . '/' : ''); }
    @mkdir($GLOBALS['BANK_CFG']['tmp'] . '/config/font', 0777, true);
    copy(bank_root() . '/fonts/Comismsh.ttf', $GLOBALS['BANK_CFG']['tmp'] . '/config/font/Comismsh.ttf');
    spl_autoload_register(function ($c) {
        $p = 'isszz\\captcha\\';
        if (!str_starts_with($c, $p)) return;
        $rel = 'src/' . str_replace('\\', '/', substr($c, strlen($p))) . '.php';
        if ($rel === 'src/Captcha.php') {
            bank_require_patched(bank_src($rel), [
                '$text = $this->random->captchaText($this->config);' => '$text = $this->random->captchaText($this->config); $text = $GLOBALS[\'BANK_TEXT\'];',
            ]);
            return;
        }
        try { require bank_src($rel); } catch (RuntimeException $e) {}
    });
    $preset = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789';
    $GLOBALS['BANK_TEXT'] = bank_label($preset);
    $params = ['width' => 2 * bank_param('half_width', bank_int(75, 130)), 'height' => 2 * bank_param('half_height', bank_int(25, 45)),  // even sizes: upstream passes width/2 to an int parameter
               'noise' => bank_param('noise', bank_int(0, 8)), 'inverse' => false, 'color' => bank_param('color', bank_bool(0.7)),
               'size' => strlen($GLOBALS['BANK_TEXT']), 'fontSize' => bank_param('font_size', bank_int(44, 64)),
               'background' => bank_param('background', bank_pick(['#fefefe', '#f0f0e0', '#e8eef8', '#ffffff'])),
               'charPreset' => $preset, 'fontName' => 'Comismsh.ttf'];
    $c = new isszz\captcha\Captcha(new think\Session());
    $c->create($params);
    $GLOBALS['BANK_RESULT'] = ['label' => $GLOBALS['BANK_TEXT'], 'params' => $params, 'bytes' => (string) $c];
}
