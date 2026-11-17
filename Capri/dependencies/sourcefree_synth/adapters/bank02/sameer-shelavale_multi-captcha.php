<?php
require bank_src('src/BaseCaptcha.php');
require bank_src('src/types/image/image.php');
$text = bank_label('abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789');
$c = new MultiCaptcha\Types\Image('secret', 1);
$c->width = bank_param('width', bank_int(150, 260));
$c->height = bank_param('height', bank_int(40, 80));
$c->noiseLevel = bank_param('noise_chars', bank_int(0, 40));
$noise = bank_param('noise_mode_chars', bank_bool(0.7));   // false switches to the coloured grid background
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $c->font = bank_font(false);
$uri = $c->getImage($text, $noise);
$GLOBALS['BANK_RESULT'] = ['label' => $text, 'bytes' => $uri,
    'params' => ['w' => $c->width, 'h' => $c->height, 'noise_chars' => $c->noiseLevel, 'noise_mode' => $noise ? 'chars' : 'grid']];
