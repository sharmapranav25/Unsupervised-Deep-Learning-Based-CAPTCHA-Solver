<?php
// Script-style library (GD bitmap font, white-on-black). Controlled text replaces random_string().
$GLOBALS['BANK_TEXT'] = bank_label('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789');
bank_require_patched(bank_src('captcha.php'), [
    '$captcha_string = random_string();' => '$captcha_string = random_string(); $captcha_string = $GLOBALS[\'BANK_TEXT\'];',
], true);
$GLOBALS['BANK_RESULT'] = ['label' => $_SESSION['captcha'] ?? $GLOBALS['BANK_TEXT'], 'params' => ['upstream_defaults' => true]];
