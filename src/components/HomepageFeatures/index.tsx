import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Complete ROS 2 Foundation',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Master the Robot Operating System 2 (ROS 2) - the middleware that connects
        all components of a robotic system. Learn about nodes, topics, services,
        actions, and Quality of Service policies for real-time constraints.
      </>
    ),
  },
  {
    title: 'Digital Twin & Simulation',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Create realistic simulation environments using Gazebo and Unity. Learn
        physics simulation, sensor modeling, and environment building for
        testing robotic systems before deployment.
      </>
    ),
  },
  {
    title: 'AI Perception & Vision-Language-Action',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Implement advanced perception systems with NVIDIA Isaac, integrate
        Vision-Language-Action paradigms, and build robots that respond to
        natural language commands using LLMs.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={styles.featureItem}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresGrid}>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
