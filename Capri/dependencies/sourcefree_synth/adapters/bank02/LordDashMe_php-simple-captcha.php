<?php
spl_autoload_register(function ($c) {
    $p = 'LordDashMe\\SimpleCaptcha\\';
    if (str_starts_with($c, $p)) require bank_src('src/' . str_replace('\\', '/', substr($c, strlen($p))) . '.php');
});
$params = [
    'font_size_min' => bank_param('font_size_min', bank_int(20, 26)), 'font_size_max' => bank_param('font_size_max', bank_int(26, 32)),
    'angle_min' => 0, 'angle_max' => bank_param('angle_max', bank_int(0, 15)), 'shadow' => bank_param('shadow', bank_bool(0.7)),
    'font_color' => bank_param('font_color', bank_pick(['#000', '#222', '#333366', '#663300'])),
];
$text = bank_label('ABCDEFGHJKLMNPRSTUVWXYZabcdefghjkmnprstuvwxyz23456789');
// chain mode: the previous stage as LordDashMe's background (path relative to its backgrounds folder)
if ($bg = bank_bg()) $params['backgrounds'] = [bank_relpath(dirname(bank_src('src/Captcha.php')) . '/../resources/backgrounds', $bg)];
$c = new LordDashMe\SimpleCaptcha\Captcha($params);
$c->code(strlen($text));                                                    // upstream draw (consumes RNG)
(new ReflectionProperty($c, 'code'))->setValue($c, $text);                  // controlled text
$c->image();
$GLOBALS['BANK_RESULT'] = ['label' => $c->getCode(), 'params' => $params, 'bytes' => $c->getImage()];
