import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'AI / Spec-Driven Humanoid Robotics',
  tagline: 'From Middleware to Vision-Language-Action',
  favicon: 'img/docusaurus.png',
  customFields: {
    description: 'A modular, simulation-first guide to building autonomous humanoid systems.',
  },

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://your-robotics-book.example.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'Farhan899', // Usually your GitHub org/user name.
  projectName: 'Ai-Robotics-Book-With-RAG-System', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  plugins: [
    // Add webpack proxy for development
    async function proxyPlugin(context, options) {
      return {
        name: 'webpack-proxy-plugin',
        configureWebpack(config, isServer, utils) {
          // Add proxy configuration for development
          config.devServer = config.devServer || {};
          config.devServer.proxy = {
            '/api': {
              target: 'http://localhost:5001', // Proxy through the Node.js server
              changeOrigin: true,
              pathRewrite: {
                '^/api': '/api', // Keep the same path structure
              },
            },
          };
          return config;
        },
      };
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Farhan899/Ai-Robotics-Book-With-RAG-System/edit/main/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'AI / Spec-Driven Humanoid Robotics',
      logo: {
        alt: 'AI Robotics Book Logo',
        src: 'img/docusaurus.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book',
        },
        {
          href: 'https://github.com/Farhan899/Ai-Robotics-Book-With-RAG-System',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'ROS 2 Module',
              to: '/docs/module-1-ros2',
            },
            {
              label: 'Digital Twin Module',
              to: '/docs/module-2-digital-twin',
            },
            {
              label: 'AI Robot Brain Module',
              to: '/docs/module-3-ai-robot-brain',
            },
            {
              label: 'VLA Module',
              to: '/docs/module-4-vla',
            },
            {
              label: 'Capstone Project',
              to: '/docs/capstone',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Docusaurus',
              href: 'https://docusaurus.io',
            },
            {
              label: 'ROS 2 Documentation',
              href: 'https://docs.ros.org/',
            },
            {
              label: 'NVIDIA Isaac',
              href: 'https://developer.nvidia.com/isaac',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Farhan899/Ai-Robotics-Book-With-RAG-System',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI Robotics Book Project. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
