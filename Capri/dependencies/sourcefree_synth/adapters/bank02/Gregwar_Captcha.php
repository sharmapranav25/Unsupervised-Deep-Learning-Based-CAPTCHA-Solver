<?php
// Gregwar/Captcha (feature bank 01) in the bank02 harness. Text is supplied; effects sampled.
// Scatter (IMG_FILTER_SCATTER) is left off: it moves pixels with C rand(), which cannot be mirrored.
spl_autoload_register(function ($c) {
    $p = 'Gregwar\\Captcha\\';
    if (str_starts_with($c, $p)) require bank_src('src/Gregwar/Captcha/' . str_replace('\\', '/', substr($c, strlen($p))) . '.php');
});
$text = bank_label('abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789');
$params = [
    'distortion' => bank_param('distortion', bank_bool(0.7)),
    'interpolation' => bank_param('interpolation', bank_bool(0.7)),
    'max_behind_lines' => bank_param('max_behind_lines', bank_int(0, 6)),
    'max_front_lines' => bank_param('max_front_lines', bank_int(0, 4)),
    'max_angle' => bank_param('max_angle', bank_int(0, 12)),
    'max_offset' => bank_param('max_offset', bank_int(0, 6)),
    'width' => bank_param('width', bank_int(140, 260)),
    'height' => bank_param('height', bank_int(40, 80)),
];
$b = new Gregwar\Captcha\CaptchaBuilder($text);
$b->setDistortion($params['distortion'])->setInterpolation($params['interpolation'])
  ->setMaxBehindLines($params['max_behind_lines'])->setMaxFrontLines($params['max_front_lines'])
  ->setMaxAngle($params['max_angle'])->setMaxOffset($params['max_offset'])->setScatterEffect(false);
if ($bg = bank_bg()) $b->setBackgroundImages([$bg]);  // chain mode: previous stage as Gregwar's background image
$b->build($params['width'], $params['height']);
ob_start(); imagepng($b->getGd()); $bytes = ob_get_clean();
$GLOBALS['BANK_RESULT'] = ['label' => $b->getPhrase(), 'bytes' => $bytes, 'params' => $params];
