import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { ChevronLeft, ChevronRight, Layers2 } from 'lucide-react';
import { Button, Tip } from './ui';

export function Inspector({
  open,
  onOpenChange,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}) {
  const id = useId();
  const toggle = useRef<HTMLButtonElement>(null);
  const origin = useRef<HTMLElement | null>(null);
  const [drawer, setDrawer] = useState(() => matchMedia('(max-width: 1180px)').matches);
  useEffect(() => {
    const query = matchMedia('(max-width: 1180px)');
    const update = () => setDrawer(query.matches);
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);
  const label = open ? '收起检查器' : '展开检查器';
  const title = (
    <span>
      <Layers2 size={16} /> 检查器
    </span>
  );
  return (
    <>
      <div className="inspector-rail">
        <Tip label={label}>
          <Button
            ref={toggle}
            className="inspector-toggle"
            variant="ghost"
            size="icon"
            aria-label={label}
            aria-expanded={open}
            aria-controls={id}
            onClick={() => onOpenChange(!open)}
          >
            {open ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </Button>
        </Tip>
      </div>
      {drawer ? (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
          <Dialog.Portal>
            <Dialog.Overlay className="dialog-overlay inspector-overlay" />
            <Dialog.Content
              id={id}
              className="inspector-pane inspector-drawer"
              aria-describedby={undefined}
              onOpenAutoFocus={() => {
                origin.current =
                  document.activeElement instanceof HTMLElement ? document.activeElement : null;
              }}
              onCloseAutoFocus={(event) => {
                event.preventDefault();
                const target = origin.current?.isConnected ? origin.current : toggle.current;
                target?.focus({ preventScroll: true });
              }}
            >
              <div className="pane-heading">
                <Dialog.Title asChild>{title}</Dialog.Title>
                <Dialog.Close asChild>
                  <Button variant="ghost" size="icon" aria-label="收起检查器">
                    <ChevronRight size={18} />
                  </Button>
                </Dialog.Close>
              </div>
              {children}
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>
      ) : (
        <aside id={id} className="inspector-pane" aria-label="详情与证据" hidden={!open}>
          <div className="pane-heading">
            {title}
            <span className="subtle-number">03</span>
          </div>
          {children}
        </aside>
      )}
    </>
  );
}
