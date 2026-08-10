import React from 'react';
import { Sparkles, UserCheck, Languages, GitBranch } from 'lucide-react';

interface SampleEssaysProps {
  onSelectSample: (text: string, label: string) => void;
}

export const SAMPLES = [
  {
    id: 'ai',
    title: 'AI Generated Essay',
    icon: <Sparkles className="w-4 h-4 text-rose-400" />,
    badge: 'AI Generated',
    badgeStyle: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    text: `I have always been fascinated by the stars. When I was five, my father bought me a small telescope. We would spend hours on the backyard deck tracing constellations and inventing names for the ones we couldn't find. Moreover, this experience taught me the value of patience and careful observation. As I reflect on those nights, I realize they shaped my academic path toward astrophysics. Furthermore, it is important to note that scientific inquiry requires a holistic and multifaceted approach. In conclusion, my journey in physics represents a tapestry of curiosity and dedication.`,
  },
  {
    id: 'human_native',
    title: 'Native Human Essay',
    icon: <UserCheck className="w-4 h-4 text-emerald-400" />,
    badge: 'Native Human',
    badgeStyle: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    text: `My hands were covered in engine grease when the 1974 Honda Civic finally sputtered to life. My uncle laughed, wiping sweat from his forehead with a rag that was dirtier than the carburetor we had just spent three Saturdays rebuilding. I wasn't thinking about college applications or career paths back then; I just wanted to figure out why the fuel pump kept vapor locking whenever the engine got hot. That rusted hatchback taught me more about mechanical thermodynamics than any textbook ever could.`,
  },
  {
    id: 'esl',
    title: 'ESL Non-Native Essay',
    icon: <Languages className="w-4 h-4 text-amber-400" />,
    badge: 'ESL Human',
    badgeStyle: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    text: `When I moved to Chicago from Seoul three years ago, English was my biggest challenge. Everyday at school I tried hard to listen to teachers and speak with classmates. I wrote down new words in my notebook every night before sleeping. Learning language is not easy, but I never gave up because I want to study business management in university. This experience made me stronger person.`,
  },
  {
    id: 'hybrid',
    title: 'Hybrid (Human + AI Edit)',
    icon: <GitBranch className="w-4 h-4 text-indigo-400" />,
    badge: 'Hybrid Essay',
    badgeStyle: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
    text: `I spent two summers volunteering at the local community garden in downtown Boston. At first, I was just pulling weeds and shoveling compost under the hot morning sun. Moreover, this experience taught me the value of community engagement and environmental stewardship. Looking back, working with urban youth fostered a profound sense of responsibility within me.`,
  },
];

export const SampleEssays: React.FC<SampleEssaysProps> = ({ onSelectSample }) => {
  return (
    <div className="space-y-2.5">
      <p className="text-xs text-slate-400 font-medium">Or try a preset sample essay:</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {SAMPLES.map((sample) => (
          <button
            key={sample.id}
            onClick={() => onSelectSample(sample.text, sample.title)}
            className="p-3 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 hover:border-slate-600 rounded-xl text-left transition-all group flex flex-col justify-between space-y-2"
          >
            <div className="flex items-center justify-between">
              {sample.icon}
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${sample.badgeStyle}`}>
                {sample.badge}
              </span>
            </div>
            <p className="text-xs font-semibold text-slate-200 group-hover:text-white transition-colors truncate">
              {sample.title}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
};
