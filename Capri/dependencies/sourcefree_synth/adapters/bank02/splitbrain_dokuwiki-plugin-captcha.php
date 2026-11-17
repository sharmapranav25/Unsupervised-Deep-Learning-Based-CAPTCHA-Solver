<?php
// DokuWiki CAPTCHA plugin (Andreas Gohr). Two independent renderers in one helper:
// imageCaptcha() = GD + bundled TTFs over line noise; svgCaptcha() = glyph paths from bundled SVG fonts.
// Upper-case A-Z like the plugin's own code generator (its SVG fonts draw some capitals as small caps).
namespace dokuwiki\Extension { class Plugin { public array $bankConf = []; public function getConf($k, $d = null) { return $this->bankConf[$k] ?? $d; } } }
namespace {
    function auth_cookiesalt($a = false, $b = false) { return 'bank'; }
    function clientIP($s = false) { return '127.0.0.1'; }
    $GLOBALS['INPUT'] = new class { public $server; public function __construct() { $this->server = $this; } public function __call($n, $a) { return ''; } };
    require bank_src('helper.php');
    $h = (new ReflectionClass('helper_plugin_captcha'))->newInstanceWithoutConstructor();
    $mode = bank_param('mode', bank_pick(['image', 'image', 'svg']));
    $text = bank_label('ABCDEFGHIJKLMNOPQRSTUVWXYZ');
    $h->bankConf = ['width' => bank_param('width', bank_int(125, 240)), 'height' => bank_param('height', bank_int(30, 70)), 'lettercount' => strlen($text)];
    $bytes = $mode === 'image' ? $h->imageCaptcha($text) : $h->svgCaptcha($text);
    $GLOBALS['BANK_RESULT'] = ['label' => $text, 'bytes' => $bytes, 'label_case' => 'insensitive',
        'params' => ['mode' => $mode] + $h->bankConf];
}
