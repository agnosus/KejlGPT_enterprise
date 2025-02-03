import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
"When should I use FatExtractor vs UniversalExtractor?",
"What are the key differences between E-916 XL and standard E-916?",
"What's the maximum pressure rating for SpeedExtractor cells?",
"Which BUCHI extraction system has the highest sample throughput?"
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
