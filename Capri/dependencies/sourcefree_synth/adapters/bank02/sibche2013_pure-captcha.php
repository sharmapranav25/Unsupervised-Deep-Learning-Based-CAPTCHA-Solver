<?php
// sibche2013/pure-captcha. Code generation (random_int, now seeded) is followed by the controlled
// text; renderImage() draws a fresh code via generate(), prints the PNG and calls exit, so the
// harness shutdown hook takes the printed image.
$charType = bank_param('char_type', bank_pick(['lettersB', 'lettersS', 'numbers']));
$alphabet = ['lettersB' => 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'lettersS' => 'abcdefghijklmnopqrstuvwxyz', 'numbers' => '0123456789'][$charType];
$GLOBALS['BANK_TEXT'] = bank_label($alphabet);
bank_require_patched(bank_src('src/Captcha.php'), [
    're:/(\$code \.= \$chars\[random_int\(0, \$maxIdx\)\];\s*\})/' => '$1 $code = $GLOBALS[\'BANK_TEXT\'];',
]);
use AminArjmand\PureCaptcha\Captcha;
$params = ['lang' => 'en', 'length' => strlen($GLOBALS['BANK_TEXT']), 'width' => bank_param('width', bank_int(150, 260)),
           'height' => bank_param('height', bank_int(50, 80)), 'font_size' => bank_param('font_size', bank_int(18, 28)),
           'char_type' => $charType,
           'noise_level' => bank_param('noise_level', bank_pick(['low', 'medium', 'high', 'extreme', 'nightmare']))];
Captcha::configure($params);
$GLOBALS['BANK_PENDING'] = ['label' => $GLOBALS['BANK_TEXT'], 'params' => $params];
Captcha::renderImage();
