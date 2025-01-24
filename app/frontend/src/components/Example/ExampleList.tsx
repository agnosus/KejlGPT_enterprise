import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
"What is the difference between K-446 and K-449?",
"Can you tell me if we have application notes about protein determination in plant meat?",
"What are the analytes that I can determine with a SpeedDigester K-439?",
"What are the analytes that I can determine with an EasyKjel?",
"Which instruments from Dist Line do have an alkali pump for NaOH dosing?",
"On which instruments is MaxAccuracy mode available?",
];

const GPT4V_EXAMPLES: string[] = [
    "Compare the impact of interest rates and GDP in financial markets.",
    "What is the expected trend for the S&P 500 index over the next five years? Compare it to the past S&P 500 performance",
    "Can you identify any correlation between oil prices and stock market trends?"
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked}: Props) => {
    const randomExamples = DEFAULT_EXAMPLES.sort(() => 0.5 - Math.random()).slice(0, 3);
    return (
        <ul className={styles.examplesNavList}>
            {randomExamples.map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
