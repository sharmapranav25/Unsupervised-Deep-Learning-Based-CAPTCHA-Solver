<?php
// Kohana 3 captcha module (Kohana Team): alpha / basic / black / word styles.
// Kohana framework touch points are minimal stand-ins; drawing code is upstream.
define('SYSPATH', bank_root() . '/');
define('MODPATH', dirname(bank_root()) . '/');
$root = bank_root();
$GLOBALS['BANK_TEXT'] = bank_label('2345679ACDEFHJKLMNPRSTUVWXYZ');
$style = bank_param('style', bank_pick(['alpha', 'basic', 'black', 'word']));
$group = [
    'style' => $style, 'width' => bank_param('width', bank_int(150, 260)), 'height' => bank_param('height', bank_int(50, 90)),
    'complexity' => bank_param('complexity', bank_int(4, 7)), 'background' => bank_bg() ?? '', 'fontpath' => $root . '/fonts/',
    'fonts' => array_map('basename', glob($root . '/fonts/*.ttf')), 'promote' => false,
];
$GLOBALS['KOHANA_CAPTCHA_CONFIG'] = (function () use ($root) { return require $root . '/config/captcha.php'; })();
$GLOBALS['KOHANA_CAPTCHA_CONFIG']['default'] = $group;

class Kohana_Exception extends Exception { public function __construct($m = '', $v = []) { parent::__construct(strtr($m, $v)); } }
class Kohana {
    public static $config;
    public static function config($path) {
        [$f, $k] = explode('.', $path, 2);
        // Controller text override also applies to the dictionary-word style.
        if ($k === 'words') return [strtolower($GLOBALS['BANK_TEXT'])];  // controlled text also for the word style
        return $GLOBALS['KOHANA_CAPTCHA_CONFIG'][$k] ?? null;
    }
}
Kohana::$config = new class { public function load($f) { return new class { public function get($g) { return $GLOBALS['KOHANA_CAPTCHA_CONFIG'][$g] ?? null; } }; } };
class Session { private static $i; private $d = []; public static function instance() { return self::$i ??= new self; } public function set($k, $v) { $this->d[$k] = $v; } public function get($k, $def = null) { return $this->d[$k] ?? $def; } public function delete($k) { unset($this->d[$k]); } }
class Request { private static $i; public $headers = []; public static function instance() { return self::$i ??= new self; } }
class url { public static function site($p = '') { return '/' . $p; } }
class utf8 {
    public static function strlen($s) { return mb_strlen($s); }
    public static function substr($s, $o, $l = null) { return mb_substr($s, $o, $l); }
    public static function strtoupper($s) { return mb_strtoupper($s); }
}
class text {
    // Kohana Text::random('distinct'): characters chosen to avoid look-alikes.
    public static function random($type = 'alnum', $length = 8) {
        // Controller text override: consume the same RNG draws as upstream, then return the chosen text.
        $pool = $type === 'distinct' ? '2345679ACDEFHJKLMNPRSTUVWXYZ' : '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
        $s = '';
        for ($i = 0; $i < $length; $i++) $s .= $pool[mt_rand(0, strlen($pool) - 1)];
        return $GLOBALS['BANK_TEXT'];  // controlled text after upstream's own draws
    }
}
require $root . '/classes/captcha.php';
require $root . '/classes/captcha/basic.php';
if ($style !== 'basic') require $root . "/classes/captcha/$style.php";
$class = 'Captcha_' . ucfirst($style);
$c = new $class('default');
ob_start(); $c->render(false); $bytes = ob_get_clean();
$label = (new ReflectionProperty('Captcha', 'response'))->getValue($c);
$GLOBALS['BANK_RESULT'] = ['label' => $label, 'bytes' => $bytes, 'label_case' => 'insensitive',
    'params' => ['style' => $style, 'w' => $group['width'], 'h' => $group['height'], 'complexity' => $group['complexity']]];
