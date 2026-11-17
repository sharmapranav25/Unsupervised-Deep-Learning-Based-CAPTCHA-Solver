<?php
// yasirmturk/simple-php-captcha (feature bank 01): original image route of the script, driven by
// its session config. Bundled background textures (rights unresolved in the pilot) are replaced
// by a generated plain background; the font comes from font bank 02.
$bg = $GLOBALS['BANK_CFG']['tmp'] . '/bg.png';
$w = bank_param('width', bank_int(180, 280)); $h = bank_param('height', bank_int(60, 100));
$im = imagecreatetruecolor($w, $h);
$c = bank_pick([[255, 255, 255], [240, 236, 220], [222, 232, 246], [236, 246, 232], [250, 240, 245]]);
imagefill($im, 0, 0, imagecolorallocate($im, ...$c));
imagepng($im, $bg);
if ($chain = bank_bg()) $bg = $chain;  // chain mode: previous stage replaces the generated plain background
$text = bank_label('ABCDEFGHJKLMNPRSTUVWXYZabcdefghjkmnprstuvwxyz23456789');
$size = bank_param('font_size', bank_int(20, 32));
$amax = bank_param('angle_max', bank_int(0, 12));
$config = ['code' => $text, 'backgrounds' => [$bg], 'fonts' => [bank_font(false)],
    'min_font_size' => $size, 'max_font_size' => $size, 'color' => bank_pick(['#666', '#223355', '#552233', '#335522', '#000']),
    'angle_min' => 0, 'angle_max' => $amax, 'shadow' => bank_param('shadow', bank_bool(0.6)),
    'shadow_color' => bank_pick(['#fff', '#aaa', '#ccc']), 'shadow_offset_x' => bank_int(-3, 3), 'shadow_offset_y' => bank_int(-3, 3)];
// The script calls session_start() itself, so the config must be in the session store, not just $_SESSION.
session_id('bank' . $GLOBALS['BANK_CFG']['seed']);
session_start();
$_SESSION['_CAPTCHA']['config'] = serialize($config);
session_write_close();
$_GET['_CAPTCHA'] = 1;
require bank_src('simple-php-captcha.php');
$GLOBALS['BANK_RESULT'] = ['label' => $text, 'params' => ['font' => basename(dirname($config['fonts'][0]))]
    + array_diff_key($config, ['code' => 1, 'backgrounds' => 1, 'fonts' => 1])];
