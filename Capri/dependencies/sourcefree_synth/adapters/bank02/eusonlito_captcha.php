<?php
// eusonlito/captcha: Captcha::source() = random string + session + new Image($string, w, h)->base64().
// The adapter performs the same steps with the controlled text (no session needed).
spl_autoload_register(function ($c) {
    $p = 'Eusonlito\\Captcha\\';
    if (str_starts_with($c, $p)) require bank_src('src/' . substr($c, strlen($p)) . '.php');
});
use Eusonlito\Captcha\{Captcha, Image};
$text = bank_label('ABCDEFGHJKLMNPRSTUVWXYZ');
$params = ['width' => bank_param('width', bank_int(180, 260)), 'height' => bank_param('height', bank_int(60, 90)),
           'noise_points' => bank_param('noise_points', bank_int(0, 80)), 'noise_lines' => bank_param('noise_lines', bank_int(0, 6)),
           'padding' => bank_param('padding', bank_float(0.2, 0.5))];
Captcha::setNoise($params['noise_points'], $params['noise_lines']);
Captcha::setPadding($params['padding']);
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') Captcha::setFont(bank_font(true));
$src = (new Image($text, $params['width'], $params['height']))->base64();
$GLOBALS['BANK_RESULT'] = ['label' => $text, 'params' => $params, 'bytes' => $src];
