<?php
// Script-style: img_gen.php draws 4 characters in a fixed 4-slot layout. Controlled text replaces
// the code; the slot loop and canvas width follow its length (identical to upstream for length 4).
// Fonts are referenced relative to the script directory (GD needs GDFONTPATH for that on Windows).
$GLOBALS['BANK_TEXT'] = bank_label('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefhijklmnopqrstuvwxyz1234567890');
$script = bank_src('captcha-generator/img_gen.php');
chdir(dirname($script));
putenv('GDFONTPATH=' . realpath(dirname($script)));
bank_require_patched($script, [
    '$text=substr($string_shuff,0,4);' => '$text=substr($string_shuff,0,4); $text=$GLOBALS[\'BANK_TEXT\'];',
    '$image_width=(32*4)+30;' => '$image_width=(32*strlen($text))+30;',
    'for ($i=1; $i<=4;$i++){' => 'for ($i=1; $i<=strlen($text);$i++){',
]);
$GLOBALS['BANK_RESULT'] = ['label' => $_SESSION['secure'] ?? null, 'params' => ['upstream_defaults' => true]];
