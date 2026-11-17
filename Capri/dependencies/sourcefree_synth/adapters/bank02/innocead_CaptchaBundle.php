<?php
// InnoceadCaptchaBundle: Symfony port of Cryptographp. Stand-ins for the Symfony Session and
// Response types it uses; drawing code is upstream. Its randomness is rand() (seeded).
// Controlled text replaces each chosen character after upstream's own draw; fonts are the bundled
// set or a per-case folder of font-bank fonts.
namespace Symfony\Component\HttpFoundation\Session { class Session { private $d = []; public function get($k, $def = null) { return $this->d[$k] ?? $def; } public function set($k, $v) { $this->d[$k] = $v; } public function has($k) { return isset($this->d[$k]); } public function remove($k) { unset($this->d[$k]); } } }
namespace Symfony\Component\HttpFoundation {
    class Response {
        public $headers; private $content = '';
        public function __construct() { $this->headers = new class { public function __call($n, $a) { return null; } }; }
        public function setContent($c) { $this->content = $c; return $this; }
        public function getContent() { return $this->content; }
        public function __call($n, $a) { return $this; }
    }
}
namespace {
    $GLOBALS['BANK_TEXT'] = bank_label('ABCDEFGHKLMNPRTWXYZ234569');
    bank_require_patched(bank_src('Captcha.php'), [
        // PHP 8 shim: brace string offsets -> brackets (three sites in getRandomCharacter()).
        're:/(\$this->config\[\'[a-z_]+\'\])\{(rand\(.*?\))\};/s' => '$1[$2];',
        // Upstream calls the alpha allocator with four arguments for opaque ink; PHP 8 requires five.
        're:/(\/\/normal\/opaque ink\s*\$ink = )imagecolorallocatealpha\(/' => '$1imagecolorallocate(',
        // per-slot font override (params.slot_fonts), after the upstream font draw
        '$this->captcha[\'letters\'][$i][\'font_path\'] = $this->config[\'char_fonts_dir\'] . DIRECTORY_SEPARATOR . $this->captcha[\'letters\'][$i][\'font\'];'
            => '$this->captcha[\'letters\'][$i][\'font_path\'] = \bank_slot_font($i - 1, $this->config[\'char_fonts_dir\'] . DIRECTORY_SEPARATOR . $this->captcha[\'letters\'][$i][\'font\']);',
        // controlled text, after the upstream character draw
        '$this->captcha[\'letters\'][$i][\'char\'] = $this->getRandomCharacter();'
            => '$this->captcha[\'letters\'][$i][\'char\'] = $this->getRandomCharacter(); $this->captcha[\'letters\'][$i][\'char\'] = $GLOBALS[\'BANK_TEXT\'][$i - 1];',
    ]);
    $fontsDir = realpath(bank_root() . '/Resources/fonts');
    if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') {
        $fontsDir = $GLOBALS['BANK_CFG']['tmp'] . '/fonts';
        @mkdir($fontsDir);
        for ($i = 0, $n = bank_param('font_count', bank_int(1, 3)); $i < $n; $i++) { $f = bank_font(true); copy($f, $fontsDir . '/' . basename(dirname($f)) . '.ttf'); }
    }
    $len = strlen($GLOBALS['BANK_TEXT']);
    $p = fn($k, $v) => bank_param($k, $v);
    $conf = [
        'width' => $p('width', bank_int(150, 260)), 'height' => $p('height', bank_int(45, 80)),
        'bg_red' => $p('bg_red', bank_int(200, 255)), 'bg_green' => $p('bg_green', bank_int(200, 255)), 'bg_blue' => $p('bg_blue', bank_int(200, 255)),
        'bg_transparent' => false, 'bg_border' => $p('bg_border', bank_bool(0.3)),
        'char_random_color' => $p('char_random_color', bank_bool(0.7)), 'char_random_color_lvl' => $p('char_random_color_lvl', bank_int(1, 3)),
        'char_transparent' => $p('char_transparent', bank_int(0, 40)), 'char_px_spacing' => $p('char_px_spacing', bank_int(22, 34)),
        'char_min_size' => $p('char_min_size', bank_int(16, 20)), 'char_max_size' => $p('char_max_size', bank_int(20, 28)),
        'char_max_rot_angle' => $p('char_max_rot_angle', bank_int(0, 35)), 'char_vertical_offset' => $p('char_vertical_offset', bank_bool()),
        'char_fonts' => array_map('basename', glob($fontsDir . '/*.ttf')), 'char_fonts_dir' => $fontsDir,
        'min_chars' => $len, 'max_chars' => $len, 'brush_size' => $p('brush_size', bank_int(1, 3)),
        'effect_blur' => $p('effect_blur', bank_bool(0.2)), 'effect_greyscale' => $p('effect_greyscale', bank_bool(0.1)),
        'noise_min_px' => $p('noise_min_px', 0), 'noise_max_px' => $p('noise_max_px', bank_pick([0, 100, 400])),
        'noise_min_lines' => $p('noise_min_lines', 0), 'noise_max_lines' => $p('noise_max_lines', bank_pick([0, 3, 8])),
        'noise_min_circles' => $p('noise_min_circles', 0), 'noise_max_circles' => $p('noise_max_circles', bank_pick([0, 3, 8])),
        'noise_color' => $p('noise_color', bank_int(1, 3)), 'noise_on_top' => $p('noise_on_top', bank_bool(0.3)),
        'test_queries_flood' => false,
    ];
    $c = new Innocead\CaptchaBundle\Captcha(new Symfony\Component\HttpFoundation\Session\Session(), $conf);
    $resp = $c->generateResponse();
    unset($conf['char_fonts_dir'], $conf['char_fonts']);
    $GLOBALS['BANK_RESULT'] = ['label' => $GLOBALS['BANK_TEXT'], 'bytes' => $resp->getContent(), 'params' => $conf];
}
