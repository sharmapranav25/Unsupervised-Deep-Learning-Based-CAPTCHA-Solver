<?php
// P-Captcha: digit code (originally from openssl random, now seeded) drawn in dotted/pixelated glyphs.
// Controlled digits replace the code after the upstream draw. Optional bank font.
$GLOBALS['BANK_TEXT'] = bank_label('0123456789');
bank_require_patched(bank_src('PCaptcha.php'), [
    '$security_code = substr(intval($num) ,0, $this->leng);' => '$security_code = substr(intval($num) ,0, $this->leng); $security_code = $GLOBALS[\'BANK_TEXT\'];',
]);
$c = new PCaptcha();
$c->width = bank_param('width', bank_int(150, 260));
$c->hight = bank_param('height', bank_int(45, 80));
$c->leng = strlen($GLOBALS['BANK_TEXT']);
$c->noise_level = bank_param('noise_level', bank_int(0, 3));
$c->angle = bank_param('angle', bank_int(0, 30));
$c->grid = bank_param('grid', bank_bool());
$c->drawcross = bank_param('cross', bank_bool(0.3));
$c->drawcircles = bank_param('circles', bank_bool());
$c->pixlized = bank_param('pixlized', bank_pick([0, 1, 2, 3]));
$c->background = bank_param('background', bank_bool(0.5) ? [0, 0, 0] : [bank_int(150, 255), bank_int(150, 255), bank_int(150, 255)]);
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $c->font = bank_font(true);
ob_start(); $c->gen_cpatcha_image(); $bytes = ob_get_clean();
$GLOBALS['BANK_RESULT'] = ['label' => (string) ($_SESSION['p_code'] ?? ''), 'bytes' => $bytes,
    'params' => ['w' => $c->width, 'h' => $c->hight, 'len' => $c->leng, 'noise' => $c->noise_level, 'angle' => $c->angle,
                 'grid' => $c->grid, 'cross' => $c->drawcross, 'circles' => $c->drawcircles, 'pixlized' => $c->pixlized]];
