<?php
// laminas/laminas-captcha Image adapter. Its validator/session/stdlib parents are other Laminas
// packages; minimal stand-ins cover the methods it overrides, and the protected generateImage()
// (upstream drawing: text, dot/line noise, two-axis sine warp) is called directly.
// Laminas ships no font: one is taken from font bank 02.
namespace Laminas\Validator {
    interface ValidatorInterface { public function isValid($value); public function getMessages(); }
    abstract class AbstractValidator implements ValidatorInterface {
        protected $options = [];
        public function __construct($options = null) { if (is_array($options)) $this->setOptions($options); }
        public function setOptions($options = []) { foreach ($options as $k => $v) { $m = 'set' . ucfirst($k); if (method_exists($this, $m)) $this->$m($v); } return $this; }
        public function getOptions() { return $this->options; }
        public function getName() { return null; }
        public function setName($name) { return $this; }
        public function getMessages() { return []; }
    }
}
namespace Laminas\Stdlib { class ErrorHandler { public static function start($l = E_WARNING) {} public static function stop($t = false) { return null; } } }
namespace {
    spl_autoload_register(function ($c) {
        $p = 'Laminas\\Captcha\\';
        if (str_starts_with($c, $p)) require bank_src('src/' . str_replace('\\', '/', substr($c, strlen($p))) . '.php');
    });
    $text = bank_label('abcdefghijkmnpqrstuvwxyz23456789');
    $font = bank_font(false);
    $w = bank_param('width', bank_int(150, 260));
    $h = bank_param('height', bank_int(50, 90));
    $opts = ['font' => $font, 'imgDir' => $GLOBALS['BANK_CFG']['tmp'] . '/', 'width' => $w, 'height' => $h,
             'fontSize' => bank_param('font_size', bank_int(22, 36)),
             'dotNoiseLevel' => bank_param('dot_noise', bank_int(0, 100)),
             'lineNoiseLevel' => bank_param('line_noise', bank_int(0, 5))];
    $c = new Laminas\Captcha\Image($opts);
    (new ReflectionMethod($c, 'generateImage'))->invoke($c, 'bank', $text);
    $GLOBALS['BANK_RESULT'] = ['label' => $text, 'bytes' => file_get_contents($opts['imgDir'] . 'bank.png'),
        'params' => ['font' => basename(dirname($font))] + array_diff_key($opts, ['font' => 1, 'imgDir' => 1])];
}
