// Radix-backed shadcn/ui composition, adapted to the workbench's semantic tokens.
import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import * as SelectPrimitive from '@radix-ui/react-select';
import * as DialogPrimitive from '@radix-ui/react-dialog';
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
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  label: string;
  disabled?: boolean;
}) {
  return (
    <SelectPrimitive.Root value={value} onValueChange={onChange} disabled={disabled}>
      <SelectPrimitive.Trigger className="select-trigger" aria-label={label}>
        <SelectPrimitive.Value />
        <SelectPrimitive.Icon>
          <ChevronDown size={14} />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content className="select-content" position="popper" sideOffset={5}>
          <SelectPrimitive.Viewport>
            {options.map((option) => (
              <SelectPrimitive.Item className="select-item" key={option.value} value={option.value}>
                <SelectPrimitive.ItemText>{option.label}</SelectPrimitive.ItemText>
                <SelectPrimitive.ItemIndicator>
                  <Check size={14} />
                </SelectPrimitive.ItemIndicator>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
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
