import { Outlet } from "react-router-dom";
import styles from "./Layout.module.css";
import { useLogin } from "../../authConfig";
import { LoginButton } from "../../components/LoginButton";
import logo from "/workspaces/buchigpt_enterprise/app/frontend/src/assets/BUCHI Logo_Green.png";

const Layout = () => {
    return (
        <div className={styles.layout}>
            <header className={styles.header} role={"banner"}>
                <div className={styles.headerContainer}>
                    <div className={styles.headerTitleContainer}>
                        <img src={logo} alt="Logo" className={styles.headerLogo} />
                        <span className={styles.headerTitle}>BUCHI GPT Enterprise</span>
                    </div>
                    <h4 className={styles.headerRightText}>ExtractGPT v1.0.0</h4>
                    {useLogin && <LoginButton />}
                </div>
            </header>
            <Outlet />
        </div>
    );
};

export default Layout;
