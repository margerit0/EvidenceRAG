// Radix-backed shadcn/ui composition, adapted to the workbench's semantic tokens.
import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import * as SelectPrimitive from '@radix-ui/react-select';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { animate, motion, useMotionValue, useTransform, type MotionStyle } from 'motion/react';
import { Check, ChevronDown, X } from 'lucide-react';
import { cn } from '../lib/utils';

const buttonVariants = cva('button', {
  variants: {
    variant: { default: 'button-primary', outline: 'button-outline', ghost: 'button-ghost' },
    size: { default: '', sm: 'button-sm', icon: 'button-icon' },
  },
  defaultVariants: { variant: 'default', size: 'default' },
});
export function Button({
  className,
  variant,
  size,
  type = 'button',
  ...props
}: React.ComponentProps<'button'> & VariantProps<typeof buttonVariants>) {
  return (
    <button {...props} type={type} className={cn(buttonVariants({ variant, size }), className)} />
  );
}
export const TooltipProvider = TooltipPrimitive.Provider;
export const DialogClose = DialogPrimitive.Close;
export function Tip({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content className="tooltip" sideOffset={7}>
          {label}
          <TooltipPrimitive.Arrow className="tooltip-arrow" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}
export const Tabs = TabsPrimitive.Root;
export function TabsList(props: React.ComponentProps<typeof TabsPrimitive.List>) {
  return <TabsPrimitive.List {...props} className={cn('tabs-list', props.className)} />;
}
export function TabsTrigger(props: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return <TabsPrimitive.Trigger {...props} className={cn('tabs-trigger', props.className)} />;
}
export function TabsContent(props: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return <TabsPrimitive.Content {...props} className={cn('tabs-content', props.className)} />;
}
export function Select({
  value,
  onChange,
  options,
  label,
  disabled,
  displayValue,
  side,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  label: string;
  disabled?: boolean;
  displayValue?: string;
  side?: 'top' | 'bottom';
}) {
  const [open, setOpen] = React.useState(false);
  const [present, setPresent] = React.useState(false);
  const trigger = React.useRef<HTMLButtonElement>(null);
  const content = React.useRef<HTMLDivElement>(null);
  const closeOnPointerDown = React.useRef(false);
  const reopening = React.useRef(false);
  const reveal = useMotionValue(0);
  const offset = useTransform(reveal, [0, 1], [1, 0]);
  const changeOpen = (next: boolean) => {
    reopening.current = next && present;
    if (next) setPresent(true);
    setOpen(next);
  };
  React.useEffect(() => {
    // Keep Radix's item registration while closed, and retain its positioned
    // surface only for the exit. One value allows reversal without a reset.
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    let playback: ReturnType<typeof animate> | undefined;
    const update = () => {
      playback?.stop();
      playback = animate(reveal, open ? 1 : 0, {
        duration: media.matches ? 0 : open ? 0.24 : 0.16,
        ease: [0.22, 1, 0.36, 1],
        onComplete: () => {
          if (!open) setPresent(false);
        },
      });
    };
    update();
    media.addEventListener('change', update);
    if (open && reopening.current) {
      content.current
        ?.querySelector<HTMLElement>('[role="option"][data-state="checked"]')
        ?.focus({ preventScroll: true });
    }
    return () => {
      playback?.stop();
      media.removeEventListener('change', update);
    };
  }, [open, reveal]);
  return (
    <SelectPrimitive.Root
      value={value}
      onValueChange={onChange}
      disabled={disabled}
      open={open}
      onOpenChange={changeOpen}
    >
      <SelectPrimitive.Trigger
        ref={trigger}
        className="select-trigger"
        aria-label={label}
        style={{ pointerEvents: present ? 'auto' : undefined }}
        onPointerDown={(event) => {
          closeOnPointerDown.current = open && event.button === 0 && !event.ctrlKey;
          if (closeOnPointerDown.current) {
            event.preventDefault();
            changeOpen(false);
          }
        }}
        onClick={(event) => {
          if (closeOnPointerDown.current) {
            event.preventDefault();
            closeOnPointerDown.current = false;
          }
        }}
      >
        <SelectPrimitive.Value>{displayValue}</SelectPrimitive.Value>
        <SelectPrimitive.Icon>
          <ChevronDown size={14} className="select-chevron" />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          ref={content}
          forceMount={present ? true : undefined}
          position="popper"
          side={side}
          sideOffset={6}
          collisionPadding={10}
          asChild
          onPointerDownOutside={(event) => {
            if (trigger.current?.contains(event.target as Node)) event.preventDefault();
          }}
        >
          <motion.div
            className="select-content"
            inert={!open}
            aria-hidden={!open || undefined}
            style={{ opacity: reveal, '--select-offset-progress': offset } as MotionStyle}
          >
            <SelectPrimitive.Viewport>
              {options.map((option) => (
                <SelectPrimitive.Item
                  className="select-item"
                  key={option.value}
                  value={option.value}
                >
                  <SelectPrimitive.ItemText>{option.label}</SelectPrimitive.ItemText>
                  <SelectPrimitive.ItemIndicator>
                    <Check size={14} />
                  </SelectPrimitive.ItemIndicator>
                </SelectPrimitive.Item>
              ))}
            </SelectPrimitive.Viewport>
          </motion.div>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}
export function Dialog({
  trigger,
  title,
  description,
  children,
  open,
  onOpenChange,
  className,
  onCloseAutoFocus,
}: {
  trigger?: React.ReactNode;
  title: string;
  description: string;
  children: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
  onCloseAutoFocus?: React.ComponentProps<typeof DialogPrimitive.Content>['onCloseAutoFocus'];
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      {trigger && <DialogPrimitive.Trigger asChild>{trigger}</DialogPrimitive.Trigger>}
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="dialog-overlay" />
        <DialogPrimitive.Content
          className={cn('dialog-content', className)}
          onCloseAutoFocus={onCloseAutoFocus}
        >
          <div className="dialog-heading">
            <DialogPrimitive.Title>{title}</DialogPrimitive.Title>
            <DialogPrimitive.Close asChild>
              <Button variant="ghost" size="icon" aria-label="关闭">
                <X size={18} />
              </Button>
            </DialogPrimitive.Close>
          </div>
          <DialogPrimitive.Description className="muted">{description}</DialogPrimitive.Description>
          {children}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
