<?php
require bank_src('Src/GIF/GIFEncoder.php');
require bank_src('Src/Captcha.php');
// Library upper-cases the code it draws, so the label alphabet is upper-case alphanumerics.
$text = bank_label('ABCDEFGHJKLMNPQRSTUVWXYZ23456789');
$c = new FC\Captcha();
$c->nums = strlen($text);
$c->width = bank_param('width', bank_int(160, 260));
$c->height = bank_param('height', bank_int(50, 80));
$c->font_size = bank_param('font_size', bank_int(24, 36));
$c->interfere_line = bank_param('interfere_line', bank_int(0, 6));
$c->is_gif = false;                       // single frame; animation handled separately
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $c->font_path = bank_font(true);
$bytes = $c->getImgContent($text);
$GLOBALS['BANK_RESULT'] = ['label' => $text, 'bytes' => $bytes, 'params' => ['w' => $c->width, 'h' => $c->height,
    'font_size' => $c->font_size, 'interfere_line' => $c->interfere_line, 'font' => basename(dirname($c->font_path)) . '/' . basename($c->font_path)]];
