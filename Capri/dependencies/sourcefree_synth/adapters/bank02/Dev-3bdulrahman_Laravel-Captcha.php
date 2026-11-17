<?php
// Laravel package with PNG (GD) and SVG back-ends. Its configured TTF fonts are not bundled, so the
// PNG path uses GD's built-in bitmap font (upstream fallback) and the SVG path names font families
// that resvg resolves from system fonts.
require __DIR__ . '/_laravel_stubs.php';
spl_autoload_register(function ($c) {
    $p = 'Dev3bdulrahman\\LaravelCaptcha\\';
    if (str_starts_with($c, $p)) require bank_src('src/' . str_replace('\\', '/', substr($c, strlen($p))) . '.php');
});
$cfg = require bank_src('config/captcha.php');
$difficulty = bank_param('difficulty', bank_pick(['easy', 'medium', 'hard']));
$cfg['image']['use_svg'] = bank_param('svg', bank_bool(0.5));
$cfg['image']['width'] = bank_param('width', bank_int(160, 260));
$cfg['image']['height'] = bank_param('height', bank_int(50, 80));
$cfg['image']['characters']['hard'] = $cfg['image']['characters']['medium'];  // keep labels alphanumeric
$GLOBALS['LARAVEL_CONFIG'] = ['captcha' => $cfg];
$g = new Dev3bdulrahman\LaravelCaptcha\Generators\ImageCaptchaGenerator($difficulty);
$data = $g->generate();
$data['value'] = bank_label($cfg['image']['characters']['medium']);
Illuminate\Support\Facades\Session::put('laravel_captcha.image', $data);
$g->image();
$GLOBALS['BANK_RESULT'] = ['label' => $data['value'], 'bytes' => $GLOBALS['BANK_BYTES'] ?? null,
    'params' => ['difficulty' => $difficulty, 'svg' => $cfg['image']['use_svg'], 'w' => $cfg['image']['width'], 'h' => $cfg['image']['height']]];
