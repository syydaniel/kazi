import React from 'react';
import {Composition} from 'remotion';
import {Main} from './Main';
import {tl} from './lib/time';

export const FPS = 60;

// 片长直接取时间表的总长，画面和配音永远一样长
export const Root: React.FC = () => (
  <Composition id="Main" component={Main} durationInFrames={Math.ceil(tl.total * FPS)} fps={FPS} width={1920} height={1080} />
);
