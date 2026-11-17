<?php
// php-quickorm/Captcha: draws 4 random characters at fixed 35 px steps on a 200x77 canvas.
// Controlled text replaces each drawn character (after upstream's own rand() draw); the loop
// and canvas width follow its length (identical to upstream for length 4). Optional bank font.
$GLOBALS['BANK_TEXT'] = bank_label('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890');
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $GLOBALS['BANK_FONT'] = bank_font(false);
bank_require_patched(bank_src('Captcha.php'), [
    '$this->font = __DIR__.\'/arial.ttf\';' => '$this->font = $GLOBALS[\'BANK_FONT\'] ?? (__DIR__.\'/arial.ttf\');',
    '$this->imageResource = imagecreate(200, 77);' => '$this->imageResource = imagecreate(max(200, 35 * (strlen($GLOBALS[\'BANK_TEXT\']) + 1) + 10), 77);',
    'for ($i = 1; $i < 5; $i++) {' => 'for ($i = 1; $i <= strlen($GLOBALS[\'BANK_TEXT\']); $i++) {',
    '$char = $this->str[rand(0, $length - 1)];' => '$char = $this->str[rand(0, $length - 1)]; $char = $GLOBALS[\'BANK_TEXT\'][$i - 1];',
]);
$level = bank_param('level', bank_int(1, 3));
$c = new Captcha($level, true);
ob_start(); $c->render(); $bytes = ob_get_clean();
$GLOBALS['BANK_RESULT'] = ['label' => $c->getCode(), 'params' => ['level' => $level], 'bytes' => $bytes];
