<?php
// lifei6671/php-captcha: create() draws `number` random characters; controlled text replaces each
// character after upstream's own draw. The label is the drawn text (upstream lower-cases getText()).
$GLOBALS['BANK_TEXT'] = bank_label('2346789abcdefghjmnpqrtuxyzABCDEFGHJMNPQRTUXYZ');
require bank_src('src/CaptchaBuilderInterface.php');
bank_require_patched(bank_src('src/CaptchaBuilder.php'), [
    '$code[$i] = $this->characters[mt_rand(0, strlen($this->characters) - 1)];'
        => '$code[$i] = $this->characters[mt_rand(0, strlen($this->characters) - 1)]; $code[$i] = $GLOBALS[\'BANK_TEXT\'][$i];',
]);
$params = ['width' => bank_param('width', bank_int(150, 260)), 'height' => bank_param('height', bank_int(40, 80)),
           'number' => strlen($GLOBALS['BANK_TEXT']),
           'line' => bank_param('line', bank_bool()), 'curve' => bank_param('curve', bank_bool()), 'noise' => bank_param('noise', bank_bool())];
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $params['fonts'] = [bank_font(false)];
$b = new Minho\Captcha\CaptchaBuilder();
$b->initialize($params);
$b->create();
ob_start(); $b->output(1); $bytes = ob_get_clean();
unset($params['fonts']);
$GLOBALS['BANK_RESULT'] = ['label' => $GLOBALS['BANK_TEXT'], 'params' => $params, 'bytes' => $bytes];
