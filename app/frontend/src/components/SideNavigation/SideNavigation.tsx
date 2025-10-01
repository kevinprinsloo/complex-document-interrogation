import React, { useState } from "react";
import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";
import styles from "./SideNavigation.module.css";

interface SideNavigationProps {
    onHistoryClick?: () => void;
    onClearChatClick?: () => void;
    onUploadClick?: () => void;
    onProcessingPanelClick?: () => void;
    onSettingsClick?: () => void;
    processingStepCount?: number;
    onWidthChange?: (width: number) => void;
}

export const SideNavigation: React.FC<SideNavigationProps> = ({
    onHistoryClick,
    onClearChatClick,
    onUploadClick,
    onProcessingPanelClick,
    onSettingsClick,
    processingStepCount = 0,
    onWidthChange
}) => {
    const { t } = useTranslation();
    const [isCollapsed, setIsCollapsed] = useState(false);

    const handleToggleCollapse = () => {
        const newCollapsedState = !isCollapsed;
        setIsCollapsed(newCollapsedState);
        if (onWidthChange) {
            onWidthChange(newCollapsedState ? 80 : 280);
        }
    };

    return (
        <div className={`${styles.sideNavigation} ${isCollapsed ? styles.collapsed : ''}`}>
            <div className={styles.navHeader}>
                <div className={styles.appBrand}>
                    <div className={styles.brandIcon}>
                        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                            <rect width="32" height="32" rx="8" fill="url(#gradient)"/>
                            <path d="M8 12h16v2H8v-2zm0 4h12v2H8v-2zm0 4h16v2H8v-2z" fill="white"/>
                            <defs>
                                <linearGradient id="gradient" x1="0" y1="0" x2="32" y2="32">
                                    <stop offset="0%" stopColor="#0078d4"/>
                                    <stop offset="100%" stopColor="#106ebe"/>
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>
                    {!isCollapsed && (
                        <div className={styles.brandText}>
                            <div className={styles.brandTitle}>Document Intelligence</div>
                            <div className={styles.brandSubtitle}>Veritas</div>
                        </div>
                    )}
                </div>
                <button className={styles.collapseButton} onClick={handleToggleCollapse}>
                    <svg className={styles.collapseIcon} width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                        {isCollapsed ? (
                            <path d="M7 3l5 5-5 5V3z"/>
                        ) : (
                            <path d="M13 3L8 8l5 5V3z"/>
                        )}
                    </svg>
                </button>
            </div>
            
            <nav className={styles.navLinks}>
                <NavLink
                    to="/"
                    className={({ isActive }) => 
                        isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
                    }
                >
                    <svg className={styles.navIcon} width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M2 4a2 2 0 012-2h12a2 2 0 012 2v12a2 2 0 01-2 2H4a2 2 0 01-2-2V4zm2 0v12h12V4H4zm2 2h8v2H6V6zm0 3h8v2H6V9zm0 3h5v2H6v-2z"/>
                    </svg>
                    {!isCollapsed && <span>{t("chat")}</span>}
                </NavLink>
                
                <NavLink
                    to="/qa"
                    className={({ isActive }) => 
                        isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
                    }
                >
                    <svg className={styles.navIcon} width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M10 2C5.58 2 2 5.58 2 10s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm1 13h-2v-2h2v2zm0-3h-2V7h2v5z"/>
                    </svg>
                    {!isCollapsed && <span>{t("qa")}</span>}
                </NavLink>
                
                <div className={styles.navDivider}></div>
                
                <button className={styles.navButton} onClick={onHistoryClick}>
                    <svg className={styles.navIcon} width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M10 2C5.58 2 2 5.58 2 10s3.58 8 8 8c1.85 0 3.55-.63 4.9-1.69L13.46 14.9C12.65 15.58 11.38 16 10 16c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6h-2l3 4 3-4h-2c0-4.42-3.58-8-8-8z"/>
                        <path d="M10.5 7H9v4l3.5 2.08.75-1.23L11 10.25V7z"/>
                    </svg>
                    {!isCollapsed && <span>History</span>}
                </button>
                
                <button className={styles.navButton} onClick={onClearChatClick}>
                    <svg className={styles.navIcon} width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M8 2a2 2 0 00-2 2v1H4a1 1 0 000 2h1v9a3 3 0 003 3h4a3 3 0 003-3V7h1a1 1 0 100-2h-2V4a2 2 0 00-2-2H8zM7 5V4h6v1H7zm1 3a1 1 0 112 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 112 0v6a1 1 0 11-2 0V8z"/>
                    </svg>
                    {!isCollapsed && <span>Clear Chat</span>}
                </button>
                
                <button className={styles.navButton} onClick={onUploadClick}>
                    <svg className={styles.navIcon} width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M10 2L6 6h3v6h2V6h3l-4-4zM4 12v4a2 2 0 002 2h8a2 2 0 002-2v-4h-2v4H6v-4H4z"/>
                    </svg>
                    {!isCollapsed && <span>Upload</span>}
                </button>
                
                <button className={styles.navButton} onClick={onProcessingPanelClick}>
                    <svg className={styles.navIcon} width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M10 2C5.58 2 2 5.58 2 10s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6-6 6z"/>
                        <circle cx="10" cy="7" r="1.5"/>
                        <circle cx="6" cy="10" r="1.5"/>
                        <circle cx="14" cy="10" r="1.5"/>
                        <circle cx="8" cy="13" r="1.5"/>
                        <circle cx="12" cy="13" r="1.5"/>
                    </svg>
                    {!isCollapsed && <span>Processing</span>}
                    {processingStepCount > 0 && (
                        <span className={`${styles.badge} ${isCollapsed ? styles.badgeCollapsed : ''}`}>{processingStepCount}</span>
                    )}
                </button>
                
                <div className={styles.navSpacer}></div>
                
                <button className={styles.navButton} onClick={onSettingsClick}>
                    <svg className={styles.navIcon} width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                        <path fillRule="evenodd" d="M8.68 2.79a1.5 1.5 0 012.64 0l.67 1.16a1.5 1.5 0 001.3.75h1.34a1.5 1.5 0 011.5 1.5v1.34a1.5 1.5 0 00.75 1.3l1.16.67a1.5 1.5 0 010 2.64l-1.16.67a1.5 1.5 0 00-.75 1.3v1.34a1.5 1.5 0 01-1.5 1.5h-1.34a1.5 1.5 0 01-1.3.75l-.67 1.16a1.5 1.5 0 01-2.64 0l-.67-1.16a1.5 1.5 0 00-1.3-.75H4.5a1.5 1.5 0 01-1.5-1.5v-1.34a1.5 1.5 0 00-.75-1.3L1.09 11.32a1.5 1.5 0 010-2.64l1.16-.67a1.5 1.5 0 00.75-1.3V5.37a1.5 1.5 0 011.5-1.5h1.34a1.5 1.5 0 001.3-.75l.67-1.16z" clipRule="evenodd"/>
                    </svg>
                    {!isCollapsed && <span>Settings</span>}
                </button>
            </nav>
        </div>
    );
};
