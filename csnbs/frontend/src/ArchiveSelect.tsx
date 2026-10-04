import { useEffect, useId, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { Check, ChevronDown } from 'lucide-react';

type Option = { value: string; label: string };
type Props = { label: string; value: string; onChange: (value: string) => void; options: Option[] };

/** Select-only combobox: focus stays on the trigger while navigating its listbox. */
export default function ArchiveSelect({ label, value, onChange, options }: Props) {
  const id = useId();
  const root = useRef<HTMLDivElement>(null);
  const menu = useRef<HTMLUListElement>(null);
  const search = useRef({ text: '', at: 0 });
  const selected = Math.max(0, options.findIndex((option) => option.value === value));
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(selected);
  const optionId = (index: number) => `${id}-option-${index}`;

  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => {
      if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener('pointerdown', dismiss);
    return () => document.removeEventListener('pointerdown', dismiss);
  }, [open]);

  useEffect(() => {
    if (open) menu.current?.children[active]?.scrollIntoView({ block: 'nearest' });
  }, [open, active]);

  function show(index = selected) {
    setActive(index);
    search.current = { text: '', at: 0 };
    setOpen(true);
  }
  function choose(index: number) {
    if (options[index]) onChange(options[index].value);
    search.current = { text: '', at: 0 };
    setOpen(false);
  }
  function keyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (!options.length) return;
    const key = event.key;
    if (key === 'Tab') { if (open) choose(active); return; }
    if (key === 'Escape') {
      if (open) { event.preventDefault(); search.current = { text: '', at: 0 }; setOpen(false); }
      return;
    }
    if (key === 'Enter' || key === ' ') {
      event.preventDefault();
      if (open) choose(active); else show();
      return;
    }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(key)) {
      event.preventDefault();
      const next = key === 'Home' ? 0 : key === 'End' ? options.length - 1 : open
        ? Math.max(0, Math.min(options.length - 1, active + (key === 'ArrowDown' ? 1 : -1))) : selected;
      if (open) setActive(next); else show(next);
      return;
    }
    if (key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      const now = Date.now();
      const previous = open && now - search.current.at < 700 ? search.current.text : '';
      const prefix = previous + key.toLocaleLowerCase();
      const repeated = [...prefix].every((character) => character === prefix[0]);
      const query = repeated ? prefix[0] : prefix;
      const start = open ? active : selected;
      const offset = previous && !repeated ? 0 : 1;
      const match = Array.from({ length: options.length }, (_, step) => (start + offset + step) % options.length)
        .find((index) => options[index].label.toLocaleLowerCase().startsWith(query));
      if (!open) show(match ?? selected);
      else if (match !== undefined) setActive(match);
      search.current = { text: prefix, at: now };
    }
  }

  return <div className="archive-select" ref={root} onBlur={(event) => {
    if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
  }}>
    <span className="archive-select-label" id={`${id}-label`}>{label}</span>
    <button type="button" className="archive-select-trigger" role="combobox"
      aria-labelledby={`${id}-label`} aria-haspopup="listbox" aria-expanded={open}
      aria-controls={open ? `${id}-menu` : undefined} aria-activedescendant={open ? optionId(active) : undefined}
      disabled={!options.length} onKeyDown={keyDown} onClick={() => open ? setOpen(false) : show()}>
      <span className="archive-select-value">{options[selected]?.label ?? 'No options'}</span>
      <ChevronDown aria-hidden="true" size={16} />
    </button>
    {open && <ul className="archive-select-menu" ref={menu} id={`${id}-menu`} role="listbox" aria-labelledby={`${id}-label`}>
      {options.map((option, index) => <li className="archive-select-option" id={optionId(index)} key={option.value}
        role="option" aria-selected={option.value === value} data-active={index === active}
        onMouseMove={() => setActive(index)} onMouseDown={(event) => event.preventDefault()} onClick={() => choose(index)}>
        <span>{option.label}</span>{option.value === value && <Check aria-hidden="true" size={15} />}
      </li>)}
    </ul>}
  </div>;
}
