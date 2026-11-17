<?php
defined('TYPO3') or die();

use SJBR\SrFreecap\Controller\AudioPlayerController;
use SJBR\SrFreecap\Controller\ImageGeneratorController;
use TYPO3\CMS\Core\Utility\ExtensionManagementUtility;
use TYPO3\CMS\Core\Utility\GeneralUtility;
use TYPO3\CMS\Extbase\Utility\ExtensionUtility;

call_user_func(
    function($extKey)
    {	
		if (isset($GLOBALS['TYPO3_CONF_VARS']['FE']['cacheHash']['excludedParameters'])) {
		    $GLOBALS['TYPO3_CONF_VARS']['FE']['cacheHash']['excludedParameters'][] = 'freecapSet';
		} else {
			$GLOBALS['TYPO3_CONF_VARS']['FE']['cacheHash']['excludedParameters'] = ['freecapSet'];
		}
		$GLOBALS['TYPO3_CONF_VARS']['FE']['cacheHash']['excludedParameters'][] = 'formatName';
		
		$extensionName = GeneralUtility::underscoredToUpperCamelCase($extKey);
		// Configuring the captcha image generator
		ExtensionUtility::configurePlugin(
			// The extension name (in UpperCamelCase) or the extension key (in lower_underscore)
			$extensionName,
			// A unique name of the plugin in UpperCamelCase
			'ImageGenerator',
			// An array holding the controller-action-combinations that are accessible
			[
				// The first controller and its first action will be the default
				ImageGeneratorController::class => 'show',
			],
			// An array of non-cachable controller-action-combinations (they must already be enabled)
			[
				ImageGeneratorController::class => 'show',
			],
			ExtensionUtility::PLUGIN_TYPE_CONTENT_ELEMENT
		);

		// Configuring the audio captcha player
		ExtensionUtility::configurePlugin(
			// The extension name (in UpperCamelCase) or the extension key (in lower_underscore)
			$extensionName,
			// A unique name of the plugin in UpperCamelCase
			'AudioPlayer',
			// An array holding the controller-action-combinations that are accessible
			[
				// The first controller and its first action will be the default
				AudioPlayerController::class => 'play',
			],
			// An array of non-cachable controller-action-combinations (they must already be enabled)
			[
				AudioPlayerController::class => 'play',
			],
			ExtensionUtility::PLUGIN_TYPE_CONTENT_ELEMENT
		);
	},
	'sr_freecap'
);