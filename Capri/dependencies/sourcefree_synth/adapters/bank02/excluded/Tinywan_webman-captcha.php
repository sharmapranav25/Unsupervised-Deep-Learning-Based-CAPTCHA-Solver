<?php
// webman plugin; framework touch points stubbed: config() and the support\Redis store.
// The drawn text is internal to base64(), so an in-memory patch exposes $generator['value'].
namespace support { class Redis { public static function __callStatic($n, $a) { return true; } } }
namespace {
    function config($key, $default = null) {
        return (require bank_src('src/config/plugin/tinywan/captcha/app.php'))['captcha'];
    }
    bank_require_patched(bank_src('src/Captcha.php'), [
        '$generator = self::generate($config);' => '$generator = self::generate($config); $GLOBALS[\'BANK_TEXT\'] = $generator[\'value\'];',
    ]);
    $len = bank_int(4, 6);
    $size = bank_int(20, 30);
    $params = ['length' => $len, 'fontSize' => $size, 'useCurve' => bank_bool(), 'useNoise' => bank_bool(),
               'useImgBg' => bank_bool(0.3),
               'imageW' => (int) ($len * $size * 1.5 + $len * $size / 2), 'imageH' => (int) ($size * 2.5),
               'bg' => [bank_int(220, 255), bank_int(220, 255), bank_int(220, 255)]];
    $out = Tinywan\Captcha\Captcha::base64($params);
    $GLOBALS['BANK_RESULT'] = ['label' => $GLOBALS['BANK_TEXT'] ?? null, 'params' => $params, 'bytes' => $out['base64'],
        'randomness' => 'text uses random_int (not replayable); drawing uses seeded mt_rand'];
}
