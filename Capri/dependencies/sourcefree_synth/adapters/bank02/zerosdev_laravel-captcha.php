<?php
// zerosdev/laravel-captcha: generate() picks each letter inside the drawing loop; the controlled
// text replaces each letter after upstream's own mt_rand draw. Optional bank font (resolved
// relative to the package's storage/fonts dir, which is how upstream builds the font path).
require __DIR__ . '/_laravel_stubs.php';
$GLOBALS['BANK_TEXT'] = bank_label('123456789abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ');
spl_autoload_register(function ($c) {
    $p = 'ZerosDev\\LaravelCaptcha\\';
    if (!str_starts_with($c, $p)) return;
    $rel = 'src/' . str_replace('\\', '/', substr($c, strlen($p))) . '.php';
    if ($rel === 'src/Captcha.php') {
        bank_require_patched(bank_src($rel), [
            '$letter = $this->chars[mt_rand(0, $len-1)];' => '$letter = $this->chars[mt_rand(0, $len-1)]; $letter = $GLOBALS[\'BANK_TEXT\'][$i];',
        ]);
        return;
    }
    require bank_src($rel);
});
$cfg = ['width' => bank_param('width', bank_int(150, 260)), 'height' => bank_param('height', bank_int(40, 80)),
        'length' => strlen($GLOBALS['BANK_TEXT'])];
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') {
    $cfg['font'] = bank_relpath(bank_root() . '/storage/fonts', bank_font(false));
}
$GLOBALS['LARAVEL_CONFIG'] = ['captcha' => $cfg];
$c = new ZerosDev\LaravelCaptcha\Captcha();
$c->generate();
$GLOBALS['BANK_RESULT'] = ['label' => $GLOBALS['BANK_TEXT'], 'bytes' => str_replace('base64, ', 'base64,', $c->image()),
    'params' => array_diff_key($cfg, ['font' => 1])];
