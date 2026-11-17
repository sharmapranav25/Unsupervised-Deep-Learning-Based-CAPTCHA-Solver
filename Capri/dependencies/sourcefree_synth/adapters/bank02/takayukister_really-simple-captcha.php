<?php
define('ABSPATH', $GLOBALS['BANK_CFG']['tmp'] . '/');  // plugin exits unless loaded inside WordPress
// Its filesystem wrapper requires WP's direct-filesystem classes from ABSPATH; provide a plain-PHP stand-in.
@mkdir(ABSPATH . 'wp-admin/includes', 0777, true);
file_put_contents(ABSPATH . 'wp-admin/includes/file.php', '<?php
function request_filesystem_credentials($u) { return true; }
function WP_Filesystem($c = false) { $GLOBALS["wp_filesystem"] = new WP_Filesystem_Direct(); return true; }
function wp_trigger_error($f, $m) { trigger_error($m); }
function __($s, $d = null) { return $s; }');
file_put_contents(ABSPATH . 'wp-admin/includes/class-wp-filesystem-base.php', '<?php class WP_Filesystem_Base {}');
file_put_contents(ABSPATH . 'wp-admin/includes/class-wp-filesystem-direct.php', '<?php
class WP_Filesystem_Direct extends WP_Filesystem_Base {
  function chmod($f, $m = false, $r = false) { return @chmod($f, $m ?: 0644); }
  function delete($f, $r = false, $t = false) { return is_file($f) ? @unlink($f) : true; }
  function get_contents($f) { return @file_get_contents($f); }
  function put_contents($f, $c, $m = false) { return file_put_contents($f, $c) !== false; }
  function exists($f) { return file_exists($f); }
  function is_dir($f) { return is_dir($f); }
  function mkdir($d, $m = false) { return @mkdir($d, 0777, true); }
  function dirlist($d) { $o = []; foreach (glob(rtrim($d, "/") . "/*") ?: [] as $p) $o[basename($p)] = ["name" => basename($p), "lastmodunix" => filemtime($p), "type" => is_dir($p) ? "d" : "f"]; return $o; }
  function mtime($f) { return @filemtime($f); }
}');
// WordPress plugin class; the handful of WP helpers it calls are provided as plain equivalents.
function wp_rand($min = 0, $max = 0) { return mt_rand($min, $max); }
function trailingslashit($s) { return rtrim($s, '/\\') . '/'; }
function path_join($base, $path) { return rtrim($base, '/\\') . '/' . ltrim($path, '/\\'); }
function wp_normalize_path($p) { return str_replace('\\', '/', $p); }
function wp_mkdir_p($d) { return is_dir($d) || mkdir($d, 0777, true); }
function wp_is_writable($p) { return is_writable($p); }
function sanitize_file_name($f) { return preg_replace('/[^A-Za-z0-9._-]/', '', $f); }
function wp_generate_password($n = 12, $s = true, $e = false) { return str_repeat('x', $n); }
require bank_src('really-simple-captcha.php');
$c = new ReallySimpleCaptcha();
$c->tmp_dir = $GLOBALS['BANK_CFG']['tmp'] . '/rsc';
$word = bank_label('ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789');
$c->char_length = strlen($word);
$scale = bank_param('scale', bank_float(1.5, 3.0));
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $c->fonts = [bank_font(false)];
$c->font_size = (int) round(14 * $scale);
$c->font_char_width = (int) round(15 * $scale);
$c->img_size = [(int) round((12 + 15 * $c->char_length) * $scale), (int) round(24 * $scale)];
$c->base = [(int) round(6 * $scale), (int) round(18 * $scale)];
$c->fg = bank_param('fg', bank_pick([[0, 0, 0], [40, 40, 120], [120, 30, 30]]));
$c->bg = bank_param('bg', bank_pick([[255, 255, 255], [240, 240, 220], [220, 235, 250]]));
$file = $c->generate_image('case', $word);
$GLOBALS['BANK_RESULT'] = ['label' => $word, 'bytes' => file_get_contents(trailingslashit($c->tmp_dir) . $file),
    'params' => ['len' => $c->char_length, 'font_size' => $c->font_size, 'img_size' => $c->img_size, 'fg' => $c->fg, 'bg' => $c->bg]];
