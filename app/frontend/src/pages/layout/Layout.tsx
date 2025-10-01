import React, { useState } from "react";
import { Outlet, Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import styles from "./Layout.module.css";

import { useLogin } from "../../authConfig";
import { SideNavigation } from "../../components/SideNavigation";
import { LoginButton } from "../../components/LoginButton";

const Layout = () => {
    const { t } = useTranslation();
    const [sidebarWidth, setSidebarWidth] = useState(280);

    const handleSidebarWidthChange = (width: number) => {
        setSidebarWidth(width);
    };

    return (
        <div className={styles.layout}>
            <SideNavigation onWidthChange={handleSidebarWidthChange} />
            <header 
                className={styles.header} 
                role={"banner"}
                style={{ marginLeft: `${sidebarWidth}px` }}
            >
                <div className={styles.headerContainer}>
                    <Link to="/" className={styles.headerTitleContainer}>
                        <h3 className={styles.headerTitle}>{t("headerTitle")}</h3>
                    </Link>
                    <div className={styles.loginMenuContainer}>
                        {useLogin && <LoginButton />}
                    </div>
                </div>
            </header>

            <main 
                className={styles.main} 
                id="main-content"
                style={{ marginLeft: `${sidebarWidth}px` }}
            >
                <Outlet />
            </main>
        </div>
    );
};

export default Layout;
