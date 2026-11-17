<?php
// Original freeCap 1.4.1 (Howard Yeend, GPL) as shipped in sjbr/sr-freecap Contrib/.
// In-memory PHP 8 shims: brace string offsets -> brackets; the microtime reseed is disabled
// so the harness seed controls mt_rand; asset paths point at the extension's bundled
// GD fonts and background images; the plaintext word is exposed (upstream stores a hash only).
$script = bank_src('Contrib/freeCap-1.4.1/freecap141.php');
$res = realpath(dirname($script) . '/../../Resources/Private/Captcha');
$fonts = [];
for ($i = 1; $i <= 5; $i++) $fonts[] = var_export("$res/Fonts/freecap_font$i.gdf", true);
$bgs = [];
for ($i = 1; $i <= 5; $i++) $bgs[] = var_export("$res/Images/freecap_im$i.jpg", true);
$GLOBALS['BANK_TEXT'] = bank_label('abcdeghklmnopqrsuvwxyz')  /* freeCap's own consonants+vowels; its GDF fonts are lower-case only */;
$bg_type = bank_param('bg_type', bank_int(0, 3));   // 0 none, 1 grid, 2 squiggles, 3 morphed image blocks
$col_type = bank_param('col_type', bank_int(0, 2));  // 0 one colour, 1 per-letter colour, 2 per-letter random
$blur_bg = bank_param('blur_bg', bank_bool(0.3));
$max_len = bank_param('max_word_length', bank_int(5, 8));  // upstream draws length from 5..max
// sendImage() prints the PNG and calls exit; the shutdown hook reads the word from the session.
$GLOBALS['BANK_PENDING'] = ['label_session' => ['bank_word'],
    'params' => ['bg_type' => $bg_type, 'col_type' => $col_type, 'blur_bg' => $blur_bg, 'max_word_length' => $max_len]];
bank_require_patched($script, [
    '$c_wid{8})+ord($c_wid{9})+ord($c_wid{10})+ord($c_wid{11})' => '$c_wid[8])+ord($c_wid[9])+ord($c_wid[10])+ord($c_wid[11] ?? "\0")',
    '$vowels{$rand_func(0,strlen($vowels)-1)}' => '$vowels[$rand_func(0,strlen($vowels)-1)]',
    '$consonants{$rand_func(0,strlen($consonants)-1)}' => '$consonants[$rand_func(0,strlen($consonants)-1)]',
    '$word{$i}' => '$word[$i]',
    '$seed_func(make_seed());' => '/* reseed disabled: harness seeds mt_rand */',
    '$use_dict = 1;' => '$use_dict = 0;',
    '$max_word_length = 6;' => '$max_word_length = ' . $max_len . ';',
    '$col_type = 1;' => '$col_type = ' . $col_type . ';',
    '$bg_type = 2;' => '$bg_type = ' . $bg_type . ';',
    '$blur_bg = false;' => '$blur_bg = ' . var_export($blur_bg, true) . ';',
    '$font_locations = Array("./.ht_freecap_font1.gdf","./.ht_freecap_font2.gdf","./.ht_freecap_font3.gdf","./.ht_freecap_font4.gdf","./.ht_freecap_font5.gdf");'
        => '$font_locations = Array(' . implode(',', $fonts) . ');',
    '$bg_images = Array("./.ht_freecap_im1.jpg","./.ht_freecap_im2.jpg","./.ht_freecap_im3.jpg","./.ht_freecap_im4.jpg","./.ht_freecap_im5.jpg");'
        => '$bg_images = Array(' . implode(',', $bgs) . ');',
    // text override applied after upstream word generation (same RNG consumption), before drawing
    "\$_SESSION['freecap_word_hash'] = \$hash_func(\$word);" => "\$word = \$GLOBALS['BANK_TEXT']; \$_SESSION['freecap_word_hash'] = \$hash_func(\$word); \$_SESSION['bank_word'] = \$word;",
], true);
