<?php
// ThinkPHP think-captcha. Stand-ins for think\Config, think\Session and the response() helper.
// The answer is hashed before storage; the in-memory patch substitutes the controlled text for the
// generated bag (after upstream's own draws) and skips the bcrypt hash.
namespace think {
    class Config { public function __construct(public array $c = []) {} public function get($k, $d = null) { return $k === 'captcha' ? $this->c : $d; } }
    class Session { private $d = []; public function set($k, $v) { $this->d[$k] = $v; } public function get($k, $d = null) { return $d; } public function has($k) { return false; } public function delete($k) {} }
}
namespace {
    function response($content = '', $code = 200, $header = []) {
        $GLOBALS['BANK_BYTES'] = $content;
        return new class { public function contentType($t) { return $this; } };
    }
    bank_require_patched(bank_src('src/Captcha.php'), [
        '$hash = password_hash($key, PASSWORD_BCRYPT, [\'cost\' => 10]);' => '$bag = $GLOBALS[\'BANK_TEXT\']; $hash = \'bank\';',
        // chain mode: upstream reads only .jpg backgrounds from its own folder (after its own pick);
        // the previous stage is given as a quality-95 JPEG copy and resampled by upstream code
        '$gb = $bgs[array_rand($bgs)];' => '$gb = $bgs[array_rand($bgs)]; if ($__chain = \bank_bg_jpeg($GLOBALS[\'BANK_CFG\'][\'tmp\'])) $gb = $__chain;',
    ]);
    $GLOBALS['BANK_TEXT'] = bank_label('2345678abcdefhijkmnpqrstuvwxyzABCDEFGHJKLMNPQRTUVWXY');
    $len = strlen($GLOBALS['BANK_TEXT']);
    $size = bank_param('font_size', bank_int(20, 30));
    $cfg = ['length' => $len, 'fontSize' => $size, 'useCurve' => bank_param('use_curve', bank_bool()),
            'useNoise' => bank_param('use_noise', bank_bool()), 'useImgBg' => bank_param('use_img_bg', bank_bool(0.3)),
            'imageW' => (int) ($len * $size * 1.5 + $len * $size / 2), 'imageH' => (int) ($size * 2.5),
            'bg' => bank_param('bg', [bank_int(220, 255), bank_int(220, 255), bank_int(220, 255)])];
    // upstream prepends its own assets/ttfs/ folder to fontttf, so a bank font is given relative to it
    if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $cfg['fontttf'] = bank_relpath(bank_root() . '/assets/ttfs', bank_font(false));
    $c = new think\captcha\Captcha(new think\Config($cfg), new think\Session());
    $c->create();
    $GLOBALS['BANK_RESULT'] = ['label' => $GLOBALS['BANK_TEXT'], 'params' => array_diff_key($cfg, ['fontttf' => 1]),
        'bytes' => $GLOBALS['BANK_BYTES'] ?? null];
}
