<?php
// dapphp/securimage (feature bank 01). Text supplied via display_value; font from font bank 02;
// no session, headers or exit. Effects: perturbation warp, curved lines, noise, text angles,
// random baseline/spacing, transparent text.
require bank_src('securimage.php');
$text = bank_label('abcdefghijkmnopqrstuvwxzyABCDEFGHJKLMNPQRSTUVWXZY23456789');
$opts = [
    'no_session' => true, 'no_exit' => true, 'send_headers' => false, 'display_value' => $text,
    'ttf_file' => bank_font(false),
    'image_width' => bank_param('width', bank_int(160, 280)), 'image_height' => bank_param('height', bank_int(50, 100)),
    'perturbation' => bank_param('perturbation', bank_pick([0, 0.3, 0.5, 0.75, 0.9])),
    'num_lines' => bank_param('num_lines', bank_int(0, 8)),
    'noise_level' => bank_param('noise_level', bank_int(0, 6)),
    'use_text_angles' => bank_param('use_text_angles', bank_bool()),
    'use_random_baseline' => bank_param('use_random_baseline', bank_bool()),
    'use_random_spaces' => bank_param('use_random_spaces', bank_bool()),
    'use_transparent_text' => bank_param('use_transparent_text', bank_bool(0.3)),
    'text_transparency_percentage' => bank_param('text_transparency', bank_int(0, 40)),
    'image_bg_color' => new Securimage_Color(bank_pick(['#ffffff', '#f4f0e6', '#e8eef8', '#202020', '#eaf6e8'])),
    'text_color' => new Securimage_Color(bank_pick(['#707070', '#202060', '#602020', '#305030', '#e0e0e0'])),
    'line_color' => new Securimage_Color(bank_pick(['#707070', '#4060a0', '#a06040', '#909090'])),
    'noise_color' => new Securimage_Color(bank_pick(['#707070', '#4060a0', '#a06040', '#909090'])),
];
$s = new Securimage($opts);
ob_start(); $s->show(bank_bg() ?? ''); $bytes = ob_get_clean();  // chain mode: Securimage's background_image argument
$GLOBALS['BANK_RESULT'] = ['label' => $text, 'bytes' => $bytes,
    'params' => array_map(fn($v) => $v instanceof Securimage_Color ? sprintf('#%02x%02x%02x', $v->r, $v->g, $v->b) : $v,
                          array_diff_key($opts, ['no_session' => 1, 'no_exit' => 1, 'send_headers' => 1, 'display_value' => 1, 'ttf_file' => 1]))];
