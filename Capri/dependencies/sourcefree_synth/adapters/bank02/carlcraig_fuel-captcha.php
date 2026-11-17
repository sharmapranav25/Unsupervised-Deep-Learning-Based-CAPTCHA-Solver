<?php
// FuelPHP captcha package, simplecaptcha driver: glyphs are cut from PNG sprite strips (no TTF),
// then sine-distorted. Stand-ins for Fuel's Config, Session and Response; drawing is upstream.
// Controlled text: a subclass runs upstream set_key() (consuming its draws), then rebuilds the key
// from the controlled text with upstream's own symbol widths, widening the canvas if needed.
namespace Fuel\Core { class Response { public $body; public function set_header($k, $v) {} } }
namespace {
    define('DS', DIRECTORY_SEPARATOR);
    define('PKGPATH', dirname(bank_root()) . DS);
    $cfg = require bank_src('config/simplecaptcha.php');
    $cfg['font_directory'] = realpath(bank_root() . '/assets/simplecaptcha') . DS;
    $GLOBALS['BANK_TEXT'] = bank_label('23456789bcdghknpqstvxyz');  // upstream symbol list minus its blacklist
    $cfg['captcha_length'] = strlen($GLOBALS['BANK_TEXT']);
    $cfg['captcha_width'] = bank_param('width', bank_int(130, 230));
    $cfg['captcha_height'] = bank_param('height', bank_int(45, 70));
    $cfg['font_height'] = bank_param('font_height', bank_int(26, 38));
    $cfg['font_smooth'] = bank_param('font_smooth', bank_bool(0.7));
    $cfg['font_gaussian_blur'] = bank_param('font_gaussian_blur', bank_bool(0.5));
    $cfg['distort'] = bank_param('distort', bank_bool(0.8));
    $cfg['distort_amplitude'] = bank_param('distort_amplitude', bank_int(5, 20));
    $cfg['distort_period'] = bank_param('distort_period', bank_int(20, 45));
    $cfg['message'] = bank_param('message', bank_bool(0.3));           // optional footer strip with a fixed message
    $cfg['font_rgba'] = bank_param('font_rgba', [bank_int(0, 80), bank_int(0, 80), bank_int(0, 120), 0]);
    $GLOBALS['FUEL_CFG'] = $cfg;
    class Config { public static function load(...$a) { return true; } public static function get($k, $d = null) { return $k === 'simplecaptcha' ? $GLOBALS['FUEL_CFG'] : ($GLOBALS['FUEL_CFG'][substr($k, 14)] ?? $d); } }
    class Session { public static $d = []; public static function set($k, $v) { self::$d[$k] = $v; } public static function get($k, $d = null) { return self::$d[$k] ?? $d; } public static function delete($k) {} }
    require bank_src('classes/driver/simplecaptcha.php');
    // Glyph scope markers around each sprite-cell copy (labels for sprite glyphs; no-op without interception).
    bank_require_patched(bank_src('classes/driver/simplecaptcha/image.php'), [
        're:/(imagecopy\(\$this->image_captcha, \$this->image_font,[^;]*;)/' => '\\BankGD\\glyph_begin((string) $data[\'symbol\']); $1 \\BankGD\\glyph_end();',
    ]);
    class BankFuelCaptcha extends Captcha\Driver_Simplecaptcha {
        protected function set_key() {
            parent::set_key();
            $by = [];
            foreach ($this->symbol_list as $s) $by[$s['symbol']] = $s;
            $this->key_array = []; $this->key_string = ''; $this->key_width = 0;
            foreach (str_split($GLOBALS['BANK_TEXT']) as $ch) {
                $this->key_array[] = $by[$ch];
                $this->key_string .= $ch;
                $this->key_width += $by[$ch]['width'];
            }
            $this->captcha_width = max($this->captcha_width, (int) ceil($this->key_width) + 10);
            return true;
        }
    }
    $d = new BankFuelCaptcha();
    ob_start(); $d->image(); $bytes = ob_get_clean();
    $GLOBALS['BANK_RESULT'] = ['label' => $GLOBALS['BANK_TEXT'], 'bytes' => $bytes, 'params' => array_intersect_key($cfg, array_flip(
        ['captcha_width', 'captcha_height', 'font_height', 'font_smooth', 'font_gaussian_blur', 'distort', 'distort_amplitude', 'distort_period', 'message']))];
}
