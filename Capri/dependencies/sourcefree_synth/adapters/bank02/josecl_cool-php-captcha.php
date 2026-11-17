<?php
// captcha.php also renders a default image at include time (relative resource paths);
// run that from the library directory and discard it, then render our own case.
// Controlled text via a subclass of the text generator (upstream draw still consumed first).
$dir = dirname(bank_src('captcha.php'));
chdir($dir);
ob_start(); require $dir . '/captcha.php'; ob_end_clean();
class BankSimpleCaptcha extends SimpleCaptcha {
    protected function GetCaptchaText() { parent::GetCaptchaText(); return $GLOBALS['BANK_TEXT']; }
}
$GLOBALS['BANK_TEXT'] = bank_label('abcdefghijkmnopqrstuvwxyz');
$c = new BankSimpleCaptcha();
$c->resourcesPath = realpath($dir . '/resources');
$c->wordsFile = '';                       // random letters, not dictionary words
$c->minWordLength = strlen($GLOBALS['BANK_TEXT']);
$c->maxWordLength = max(strlen($GLOBALS['BANK_TEXT']), 5);  // font scaling uses maxWordLength
$c->difficulty = bank_param('difficulty', bank_float(0.3, 2.0));
$c->lineWidth = bank_param('line_width', bank_pick([0, 0, 2, 3]));
$c->blur = bank_param('blur', bank_bool(0.3));
$c->maxRotation = bank_param('max_rotation', bank_int(0, 12));
$c->Yamplitude = bank_param('y_amplitude', bank_int(4, 16));
$c->Xamplitude = bank_param('x_amplitude', bank_int(2, 8));
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') {
    // WriteText() opens resourcesPath/fonts/<file>: point at the bank font via a relative path
    $font = bank_font(false);
    $c->fonts = ['Bank' => ['spacing' => bank_param('font_spacing', bank_float(-3, 0)), 'minSize' => 24, 'maxSize' => 32,
                            'font' => bank_relpath($c->resourcesPath . '/fonts', $font)]];
}
$c->imageFormat = 'png';
ob_start(); $c->CreateImage(); $bytes = ob_get_clean();
$GLOBALS['BANK_RESULT'] = ['label' => $_SESSION['captcha'] ?? null, 'bytes' => $bytes, 'params' => [
    'difficulty' => $c->difficulty, 'lineWidth' => $c->lineWidth, 'blur' => $c->blur, 'maxRotation' => $c->maxRotation,
    'Yamplitude' => $c->Yamplitude, 'Xamplitude' => $c->Xamplitude]];
