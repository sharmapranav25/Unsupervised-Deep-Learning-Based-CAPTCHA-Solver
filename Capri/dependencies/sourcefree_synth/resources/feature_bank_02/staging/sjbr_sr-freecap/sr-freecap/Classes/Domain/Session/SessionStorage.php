<?php
namespace SJBR\SrFreecap\Domain\Session;

/*
 *  Copyright notice
 *
 *  (c) 2012-2026 Stanislas Rolland <typo3AAAA(arobas)sjbr.ca>
 *  All rights reserved
 *
 *  This script is part of the TYPO3 project. The TYPO3 project is
 *  free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  The GNU General Public License can be found at
 *  http://www.gnu.org/copyleft/gpl.html.
 *  A copy is found in the textfile GPL.txt and important notices to the license
 *  from the author is found in LICENSE.txt distributed with these scripts.
 *
 *
 *  This script is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  This copyright notice MUST APPEAR in all copies of the script!
 */

use Psr\Http\Message\ServerRequestInterface;
use TYPO3\CMS\Core\Session\Backend\Exception\SessionNotFoundException;
use TYPO3\CMS\Core\SingletonInterface;
use TYPO3\CMS\Frontend\Authentication\FrontendUserAuthentication;

/**
 * Session storage
 */
class SessionStorage implements SingletonInterface
{
	const SESSIONNAMESPACE = 'tx_srfreecap';

	/**
	 * Request
	 *
	 * @var ServerRequestInterface
	 */
	protected $request;
 
	/**
	 * Returns the object stored in the user's PHP session
	 *
	 * @return Object the stored object
	 */
	public function restoreFromSession()
	{
		$sessionData = $this->getUser()->getSessionData(self::SESSIONNAMESPACE);
		if ($sessionData === null) {
			return null;
		}
		return unserialize($sessionData);
	}
 
	/**
	 * Writes an object into the PHP session
	 *
	 * @param mixed any serializable object to store into the session
	 * @return SessionStorage
	 */
	public function writeToSession($object)
	{
		$sessionData = serialize($object);
		$this->getUser()->setAndSaveSessionData(self::SESSIONNAMESPACE, $sessionData);
		return $this;
	}
 
	/**
	 * Cleans up the session: removes the stored object from the PHP session
	 *
	 * @return SessionStorage
	 */
	public function cleanUpSession()
	{
		$this->getUser()->setAndSaveSessionData(self::SESSIONNAMESPACE, null);
		return $this;
	}

	/**
	 * Gets a frontend user session
	 *
	 * @return User The current frontend user object
	 */
	protected function getUser() : FrontendUserAuthentication
	{
		$feUser = $this->getRequest()->getAttribute('frontend.user') ?? false;
		if (!$feUser) {
			throw new SessionNotFoundException('No frontend user found in session!', 4303244028);
		}
		return $feUser;
	}

    /**
     * @return ServerRequestInterface
     */
	public function setRequest(ServerRequestInterface $request): void
	{
		$this->request = $request;
	}

    /**
     * @return ServerRequestInterface
     */
	private function getRequest(): ServerRequestInterface
	{
		return $this->request;
	}
}