<?php
// bonecms/laravel-captcha: the wave generator is framework-free; its Laravel config file only
// needs base_path() to locate the bundled font.
function base_path($p = '') { return bank_root() . '/' . preg_replace('#^vendor/bonecms/laravel-captcha/#', '', $p); }
foreach (['Generator/GeneratorInterface', 'Generator/AbstractGenerator', 'Generator/GeneratorWaves', 'Code/CodeInterface', 'Code/SimpleCode'] as $f) {
    require bank_src("src/Captcha/$f.php");
}
$params = require bank_src('src/config/bone/captcha.php');
// The package facade wraps these as lists; the generator indexes them as arrays.
$params['background'] = [bank_param('background', bank_pick(['f2f2f2', 'ffffff', 'eef3e2', 'f7ecd9', 'e6eef8']))];
$params['colors'] = [bank_param('colour', bank_pick(['2980b9', '1d1d1d', 'c0392b', '27ae60', '8e44ad']))];
$params['fontSize'] = bank_param('font_size', bank_int(22, 32));
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $params['font'] = bank_font(false);
$params['letterSpacing'] = bank_param('letter_spacing', bank_int(0, 8));
$code = bank_label('QSFHTRPAJKLMZXCVBNabdefhxktyzj23456789');
$len = strlen($code);
$params['width'] = bank_param('width', (int) ($len * ($params['fontSize'] * 0.66 + $params['letterSpacing']) + bank_param('margin', bank_int(30, 80))));
$params['height'] = (int) ($params['fontSize'] * bank_param('height_ratio', bank_float(1.8, 2.6)));
$bytes = (new Igoshev\Captcha\Captcha\Generator\GeneratorWaves())->render($code, $params);
unset($params['middleware'], $params['routes']);
$GLOBALS['BANK_RESULT'] = ['label' => $code, 'params' => $params, 'bytes' => $bytes];
