<?php
// Math CAPTCHA: the image shows "a + b = " in GD bitmap font; the label is the drawn expression.
require bank_src('src/MathCaptcha/MathCaptchaException.php');
require bank_src('src/MathCaptcha/MathCaptcha.php');
$c = new MathCaptcha\MathCaptcha(0);
$c->generate();
$a = (new ReflectionProperty(MathCaptcha\MathCaptcha::class, 'addNum1'))->getValue($c);
$b = (new ReflectionProperty(MathCaptcha\MathCaptcha::class, 'addNum2'))->getValue($c);
ob_start(); $c->output(); $bytes = ob_get_clean();
$GLOBALS['BANK_RESULT'] = ['label' => "$a+$b=", 'shown_text' => "$a + $b = ", 'text_kind' => 'math',
    'params' => ['answer' => $a + $b], 'bytes' => $bytes];
