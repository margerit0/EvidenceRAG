import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { AnimatePresence, motion, useIsPresent } from 'motion/react';
import { ChevronLeft, ChevronRight, Layers2 } from 'lucide-react';
import { Button, Tip } from './ui';

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => matchMedia(query).matches);
  useEffect(() => {
    const media = matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, [query]);
  return matches;
}

// Keep the same surface during an interrupted exit; Motion retargets its current
// position instead of starting another entrance from the edge of the screen.
function DrawerSurface({ children, ...props }: React.ComponentProps<typeof Dialog.Content>) {
  const present = useIsPresent();
  const reduced = useMediaQuery('(prefers-reduced-motion: reduce)');
  return (
    <Dialog.Content {...props} forceMount asChild>
      <motion.aside
        inert={!present}
        aria-hidden={!present || undefined}
        initial={{ x: reduced ? 0 : '100%', opacity: reduced ? 0 : 1 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: reduced ? 0 : '100%', opacity: reduced ? 0 : 1 }}
        transition={{ duration: reduced ? 0 : 0.38, ease: [0.22, 1, 0.36, 1] }}
        style={{ pointerEvents: present ? 'auto' : 'none' }}
      >
        {children}
      </motion.aside>
    </Dialog.Content>
  );
}

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
  const drawerClose = useRef<HTMLButtonElement>(null);
  const origin = useRef<HTMLElement | null>(null);
  const reduced = useMediaQuery('(prefers-reduced-motion: reduce)');
  const drawer = useMediaQuery('(max-width: 1180px)');
  useEffect(() => {
    // A reversed exit keeps the Dialog mounted, so its mount autofocus does not
    // run again. Return keyboard focus to the reopened modal in that case too.
    if (drawer && open) drawerClose.current?.focus({ preventScroll: true });
  }, [drawer, open]);
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
            <ChevronLeft size={18} className="inspector-chevron" />
          </Button>
        </Tip>
      </div>
      {drawer ? (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
          <AnimatePresence>
            {open && (
              <Dialog.Portal forceMount>
                <Dialog.Overlay asChild forceMount>
                  <motion.div
                    className="dialog-overlay inspector-overlay"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: reduced ? 0 : 0.28 }}
                  />
                </Dialog.Overlay>
                <DrawerSurface
                  id={id}
                  className="inspector-pane inspector-drawer"
                  aria-describedby={undefined}
                  onPointerDownOutside={(event) => {
                    // The rail can reopen an exiting surface. Do not let the
                    // deferred outside-dismiss event close it again after click.
                    if (toggle.current?.contains(event.target as Node)) event.preventDefault();
                  }}
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
                      <Button ref={drawerClose} variant="ghost" size="icon" aria-label="收起检查器">
                        <ChevronRight size={18} />
                      </Button>
                    </Dialog.Close>
                  </div>
                  {children}
                </DrawerSurface>
              </Dialog.Portal>
            )}
          </AnimatePresence>
        </Dialog.Root>
      ) : (
        <div className="inspector-slot">
          <aside
            id={id}
            className="inspector-pane inspector-desktop"
            aria-label="详情与证据"
            inert={!open}
            aria-hidden={!open}
          >
            <div className="pane-heading">
              {title}
              <span className="subtle-number">03</span>
            </div>
            {children}
          </aside>
        </div>
      )}
    </>
  );
}
