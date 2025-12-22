import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Start Reading
          </Link>
          <Link
            className="button button--outline button--primary button--lg"
            to="/docs/module-1-ros2">
            View Architecture
          </Link>
        </div>
      </div>
    </header>
  );
}

function ConceptualArchitecture() {
  return (
    <section className={styles.architectureSection}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset--2">
            <div className="text--center padding-horiz--md">
              <Heading as="h2" className={styles.sectionTitle}>
                System Architecture
              </Heading>
              <p className="padding-horiz--lg">
                The complete stack of humanoid robotics development
              </p>
            </div>
          </div>
        </div>
        <div className="row padding-vert--lg">
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <div className={styles.iconCircle}>1</div>
              <h3>ROS 2</h3>
              <p>
                Robot Operating System 2 as the middleware connecting all components
              </p>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <div className={styles.iconCircle}>2</div>
              <h3>Digital Twin</h3>
              <p>
                Physics simulation with Gazebo and Unity for safe testing
              </p>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <div className={styles.iconCircle}>3</div>
              <h3>AI Brain</h3>
              <p>
                Advanced perception systems using NVIDIA Isaac
              </p>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <div className={styles.iconCircle}>4</div>
              <h3>Vision-Language-Action</h3>
              <p>
                Natural language interfaces for human-robot interaction
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function TargetAudience() {
  return (
    <section className={styles.audienceSection}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset--2">
            <div className="text--center padding-horiz--md">
              <Heading as="h2" className={styles.sectionTitle}>
                Who This Book Is For
              </Heading>
            </div>
          </div>
        </div>
        <div className="row padding-vert--lg">
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <h3>Robotics Engineers</h3>
              <p>
                Building robotic systems with proper middleware architecture
              </p>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <h3>AI Researchers</h3>
              <p>
                Moving into embodied intelligence and perception systems
              </p>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <h3>Students</h3>
              <p>
                Learning modern robotics stacks and simulation-first approaches
              </p>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <h3>Simulation Developers</h3>
              <p>
                Creating hardware-agnostic development environments
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function WhatYouWillBuild() {
  return (
    <section className={styles.buildSection}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset--2">
            <div className="text--center padding-horiz--md">
              <Heading as="h2" className={styles.sectionTitle}>
                What You Will Build
              </Heading>
              <ul className={styles.outcomeList}>
                <li>A simulated humanoid robot</li>
                <li>A physics-based digital twin</li>
                <li>A perception and navigation stack</li>
                <li>A language-driven action planner</li>
              </ul>
              <p className={styles.buildNote}>
                <em>Simulation-first, hardware-agnostic approach</em>
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function NavigationEntryPoints() {
  return (
    <section className={styles.navSection}>
      <div className="container">
        <div className="row">
          <div className="col col--8 col--offset--2">
            <div className="text--center padding-horiz--md">
              <Heading as="h2" className={styles.sectionTitle}>
                Start Your Journey
              </Heading>
            </div>
          </div>
        </div>
        <div className="row padding-vert--lg">
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <Link className={styles.navLink} to="/docs/module-1-ros2">
                <h3>Module 1: ROS 2</h3>
                <p>Begin with middleware fundamentals</p>
              </Link>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <Link className={styles.navLink} to="/docs/capstone">
                <h3>Capstone Project</h3>
                <p>Jump to the complete system</p>
              </Link>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <Link className={styles.navLink} to="/docs/module-1-ros2/architecture">
                <h3>System Architecture</h3>
                <p>View complete technical design</p>
              </Link>
            </div>
          </div>
          <div className="col col--3">
            <div className="text--center padding-horiz--md">
              <Link className={styles.navLink} to="/docs/module-1-ros2/code-examples">
                <h3>Code Examples</h3>
                <p>Browse implementation details</p>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`AI / Spec-Driven Humanoid Robotics`}
      description="Complete guide to building intelligent humanoid robotic systems, from ROS 2 middleware to Vision-Language-Action systems">
      <HomepageHeader />
      <main>
        <ConceptualArchitecture />
        <TargetAudience />
        <WhatYouWillBuild />
        <NavigationEntryPoints />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
